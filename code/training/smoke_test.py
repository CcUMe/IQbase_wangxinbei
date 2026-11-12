"""Run the preliminary pipeline on real data, without a Docker/container build."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from utils.common import read_json, write_json
from utils.run_directory import latest_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/mnt/disk1/wangxinbei_contest_dataset/C01'))
    p.add_argument('--output', type=Path, default=Path('artifacts/quickstart'))
    p.add_argument('--prepared-data', type=Path, help='Reuse an already verified small prepared cache')
    p.add_argument('--manifest', type=Path, default=Path('artifacts/c01_data_audit/sample_manifest.parquet'))
    p.add_argument('--device', default='auto')
    p.add_argument('--gpu', default='0', help='Visible GPU for short single-GPU checks')
    p.add_argument('--ddp-gpus', choices=['0,1', '2,3', '4,5'], help='One supported, allocated communication pair')
    p.add_argument('--resume', action='store_true', help='Skip already passed stages in this output directory')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error('Choose a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    logdir = args.output / 'logs'
    logdir.mkdir(exist_ok=True)
    previous = {}
    if args.resume:
        status = read_json(args.output / 'smoke_status.json')
        previous = {stage['name']: stage for stage in status['stages']}
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.gpu
    python = sys.executable
    stages = []
    def run(name, command, env=None):
        if name in previous and previous[name]['returncode'] == 0 and name not in ['core_checks', 'config_checks', 'stability_checks', 'reporting_checks', 'contrastive_checks', 'hypothesis_checks', 'resume_checks', 'command_config_checks', 'finetuning_checks']:
            stages.append(previous[name])
            print(f'[{name}] reused previous passed stage', flush=True)
            return
        start = time.monotonic()
        print(f'[{name}] {shlex.join(map(str, command))}', flush=True)
        path = logdir / f'{name}.log'
        if path.exists():
            path.replace(logdir / f'{name}.previous.log')
        with path.open('w') as log:
            completed = subprocess.run(list(map(str, command)), cwd=ROOT, env=env or environment, stdout=log, stderr=subprocess.STDOUT)
        record = {'name': name, 'returncode': completed.returncode, 'seconds': time.monotonic() - start,
                  'command': list(map(str, command)), 'visible_gpus': (env or environment).get('CUDA_VISIBLE_DEVICES')}
        stages.append(record)
        write_json(args.output / 'smoke_status.json', {'stages': stages, 'complete': False})
        if completed.returncode:
            print((logdir / f'{name}.log').read_text()[-10000:])
            raise SystemExit(f'{name} failed; inspect {logdir / (name + ".log")}')
        print(f'[{name}] passed in {record["seconds"]:.2f}s', flush=True)

    run('core_checks', [python, 'code/training/test_core.py'])
    run('config_checks', [python, 'code/training/test_config.py'])
    run('stability_checks', [python, 'code/training/test_stability.py'])
    run('reporting_checks', [python, 'code/training/test_reporting.py'])
    run('contrastive_checks', [python, 'code/training/test_contrastive.py'])
    run('hypothesis_checks', [python, 'code/training/test_hypotheses.py'])
    run('resume_checks', [python, 'code/training/test_resume.py'])
    run('command_config_checks', [python, 'code/training/test_command_config.py'])
    run('finetuning_checks', [python, 'code/training/test_finetuning.py'])
    data = args.prepared_data.resolve() if args.prepared_data else args.output / 'data'
    if not args.prepared_data:
        if not args.manifest.exists():
            run('audit', [python, 'code/utils/audit_c01_dataset.py', '--data-root', args.data_root,
                          '--output-dir', args.manifest.parent])
        run('prepare', [python, 'code/training/prepare_data.py', '--data-root', args.data_root,
                        '--manifest', args.manifest.resolve(), '--output', data, '--limit-per-task', '96'])
    common = ['--data', data, '--device', args.device]
    small = ROOT / 'code/configs/smoke/supervised.json'
    baseline = args.output / 'supervised'
    run('supervised', [python, 'code/training/train.py', *common, '--config', small, '--output', baseline, '--max-steps', '3', '--merge-paired'])
    baseline = latest_run(baseline)
    run('resume', [python, 'code/training/train.py', *common, '--config', small, '--output', baseline,
                   '--resume', baseline / 'last.pt', '--max-steps', '6', '--merge-paired'])
    structure = read_json(ROOT / 'code/configs/smoke/pretrain.json')
    reconstruction = deepcopy(structure)
    reconstruction['loss']['pretraining']['weights'].update(reconstruction=1.0, atomic=0.0, emergent=0.0)
    recon_config = args.output / 'reconstruction_config.json'
    write_json(recon_config, reconstruction)
    run('reconstruction', [python, 'code/training/train.py', *common, '--config', recon_config,
                          '--output', args.output / 'reconstruction', '--stage', 'pretrain', '--max-steps', '3'])
    structure['loss']['pretraining']['weights']['reconstruction'] = 0.0
    pure_config = args.output / 'structure_config.json'
    write_json(pure_config, structure)
    vocabulary_common = ['--data', data, '--device', args.device, '--atomic-vocab', '32', '--merges', '32',
                          '--max-patches', '1024', '--bpe-samples', '128', '--cluster-iterations', '2', '--batch-size', '16']
    v1, v2 = args.output / 'codebook_v1', args.output / 'codebook_v2'
    run('codebook_v1', [python, 'code/training/build_codebook.py', *vocabulary_common, '--output', v1])
    stage1, stage2 = args.output / 'structure_v1', args.output / 'structure_v2'
    run('structure_v1', [python, 'code/training/train.py', *common, '--config', pure_config, '--output', stage1,
                         '--stage', 'pretrain', '--codebook', v1, '--max-steps', '3'])
    stage1 = latest_run(stage1)
    run('codebook_v2', [python, 'code/training/build_codebook.py', *vocabulary_common, '--output', v2,
                        '--teacher-checkpoint', stage1 / 'best.pt'])
    run('structure_v2_hybrid', [python, 'code/training/train.py', *common, '--config', ROOT / 'code/configs/smoke/pretrain.json',
             '--output', stage2, '--stage', 'pretrain', '--codebook', v2, '--init-checkpoint', stage1 / 'best.pt', '--max-steps', '3'])
    stage2 = latest_run(stage2)
    final = args.output / 'finetuned'
    run('finetune', [python, 'code/training/train.py', *common, '--config', small, '--output', final,
                     '--init-checkpoint', stage2 / 'best.pt', '--max-steps', '6'])
    final = latest_run(final)
    run('evaluate', [python, 'code/training/evaluate.py', *common, '--checkpoint', final / 'best.pt',
                     '--output', args.output / 'evaluation.json', '--batch-size', '16'])
    weights = args.output / 'weights'
    run('export', [python, 'code/training/export.py', '--checkpoint', final / 'best.pt', '--output', weights])
    sample = args.data_root / 'rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet'
    labels = args.data_root / 'rf_signal_benchmark_train_v1.0/task_schema.json'
    for kind, device in [('device', args.device), ('cpu', 'cpu')]:
        prediction = args.output / f'prediction_{kind}.jsonl'
        run(f'infer_{kind}', [python, 'code/infer.py', '--input', sample, '--output', prediction, '--weights', weights,
                              '--device', device, '--batch-size', '4'])
        run(f'validate_{kind}', [python, 'code/utils/validate_output.py', '--input', sample,
                                '--output', prediction, '--schema', labels])
    if args.ddp_gpus:
        ddp_env = {**environment, 'CUDA_VISIBLE_DEVICES': args.ddp_gpus}
        command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
             'code/training/train.py', '--data', data, '--config', small, '--output', args.output / 'ddp', '--max-steps', '3']
        if (args.output / 'ddp/latest_run.json').exists() or (args.output / 'ddp/last.pt').exists():
            command[-1] = '6'
            command += ['--resume', latest_run(args.output / 'ddp') / 'last.pt']
        run('ddp_two_gpus', command, ddp_env)
    report = {'complete': True, 'stages': stages, 'dataset': str(data), 'weights': str(weights),
              'note': 'Tiny model and short runs verify execution only. Metrics are not competition performance.'}
    write_json(args.output / 'smoke_status.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Real tiny-cache round1 -> rebuilt vocabulary -> round2 -> fine -> inference.

CPU execution/weight-transfer checks, never evidence of accuracy improvement.
Writes all fixtures/logs under artifacts/iterative_pipeline_checks/<timestamp>.
"""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import torch
from models.rf_model import RFModel
from training.iterative_pretrain_pipeline import read_settings, stamp, validate_devices, validate_vocabulary
from training.prepare_patch_view import prepare_view
from utils.common import model_bundle, read_json, write_json
from utils.config import load_config, model_options
from utils.initialization import load_pretrained_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processes', type=int, choices=[1, 2], default=1)
    parser.add_argument('--patch-size', type=int, choices=[8, 16], default=16)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = ROOT / 'artifacts/iterative_pipeline_checks' / stamp()
    output.mkdir(parents=True)
    fixture = read_json(ROOT / 'artifacts/shared_base_pipeline/integration_latest.json')
    data = Path(fixture['directory']) / 'data'
    if not data.is_dir():
        raise FileNotFoundError('Run the shared-base real-data fixture preparation first: ' + str(data))
    if args.patch_size==8:
        source=data
        data=output/'data_p8'
        prepare_view(source,data,8)
        assert (data/'iq.npy').resolve()==(source/'iq.npy').resolve()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}

    def run(name, command):
        with (output / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    first_book = output / 'raw_iq_vocabulary'
    run('raw_vocabulary', ['code/training/build_codebook.py', '--data', str(data),
        '--output', str(first_book), '--device', 'cpu', '--atomic-vocab', '8',
        '--merges', '4', '--max-patches', '128', '--bpe-samples', '32',
        '--cluster-iterations', '2', '--batch-size', '16', '--threads', '2'])
    validate_vocabulary(first_book, data)
    try:
        validate_vocabulary(first_book, data, output / 'unrelated_teacher.pt')
    except ValueError:
        pass
    else:
        raise AssertionError('Round2 accepted a raw-IQ/incorrect teacher vocabulary')
    for stage in ['pretrain', 'finetune']:
        basename='shared_base_p8.json5' if args.patch_size==8 else 'shared_base.json5'
        config = load_config(ROOT / 'code/configs' / stage / basename)
        config['model']['encoder'].update(width=64, depth=2, heads=4, dropout=0., attention_dropout=0.)
        config['model']['iq_geometry']['bottleneck'] = 8
        config['run'].update(data=str(data), codebook=str(first_book) if stage == 'pretrain' else '',
                             init_checkpoint='', warmstart_checkpoint='')
        config['training']['epochs'] = 1
        config['data'].update(batch_size=4, steps_per_epoch=3)
        config['data']['loader'].update(num_workers=0, pin_memory=False)
        config['runtime'].update(device='cpu', gpus='', precision='fp32', threads=2, log_every=1)
        config['checkpoint']['save_every_steps'] = 1
        config['evaluation']['batch_size'] = 16
        write_json(output / (stage + '.json'), config)
    single = read_settings(ROOT / 'code/configs/pipeline/shared_base_iterative.json5')
    assert single['runtime']['processes'] == 1 and single['data']['batch_size_per_gpu'] == 0
    for pair in ['0,1', '2,3', '4,5']:
        assert len(validate_devices(pair, 2, 'auto')) == 2
    for selection, processes in [('0,2', 2), ('0,1,2', 2), ('0,0', 2), ('0', 2), ('0,1', 1)]:
        try:
            validate_devices(selection, processes, 'auto')
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted invalid GPU/process selection')
    entry = 'iterative_pretrain_dual.py' if args.processes == 2 else 'iterative_pretrain_pipeline.py'
    name = 'shared_base_iterative_dual.json5' if args.processes == 2 else 'shared_base_iterative.json5'
    if args.patch_size==8:
        name='shared_base_iterative_p8_dual.json5'
        if args.processes!=2:raise ValueError('P8 full-chain check uses the dual pipeline')
    settings = read_settings(ROOT / 'code/configs/pipeline' / name)
    if args.processes == 2:
        assert settings['data']['batch_size_per_gpu'] * args.processes == 256
        settings['data']['batch_size_per_gpu'] = 4
    settings['configs'] = {stage: str(output / (stage + '.json')) for stage in ['pretrain', 'finetune']}
    settings['run']['output'] = str(output / 'pipeline')
    settings['round1'].update(epochs=1, codebook=str(first_book))
    settings['round2']['epochs'] = 1
    settings['codebook'].update(atomic_vocab=8, merges=4, max_patches=128,
                                cluster_iterations=2, fit_sequences=32, batch_size=16, threads=2)
    settings['runtime']['device'] = 'cpu'
    write_json(output / 'pipeline.json', settings)
    run('pipeline', ['code/training/' + entry, '--config', str(output / 'pipeline.json')])
    directory = Path(read_json(output / 'pipeline/latest_pipeline.json')['pipeline_directory'])
    assert (directory / 'completion.json').is_file()
    manifest = read_json(directory / 'manifest.json')
    stages = manifest['stages']
    assert manifest['processes'] == args.processes
    for name in ['pretrain_round1', 'pretrain_round2', 'finetune']:
        assert stages[name]['complete'] and stages[name]['global_step'] == 3
        saved = torch.load(Path(stages[name]['run_directory']) / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['world_size'] == args.processes
        assert saved['training_config']['runtime']['processes'] == args.processes
    teacher = stages['pretrain_round1']['best_checkpoint']
    validate_vocabulary(stages['codebook_round2']['output'], data, teacher)
    assert stages['pretrain_round2']['initialization'] == teacher
    assert stages['pretrain_round2']['codebook'] == stages['codebook_round2']['output']
    assert stages['finetune']['initialization'] == stages['pretrain_round2']['best_checkpoint']

    # Same-shaped structure heads must also stay freshly initialized: copying
    # an old classifier is incorrect after the cluster IDs have been reassigned.
    first = torch.load(teacher, map_location='cpu', weights_only=True)
    torch.manual_seed(987)
    second = RFModel(**model_bundle(first))
    head_before = {k: v.clone() for k, v in second.state_dict().items() if not k.startswith('backbone.')}
    load_pretrained_backbone(second, first)
    for key, value in second.state_dict().items():
        assert torch.equal(value, first['model'][key] if key.startswith('backbone.') else head_before[key]), key
    assert not torch.equal(second.aux['atomic'].weight, first['model']['aux.atomic.weight'])
    assert not torch.equal(second.aux['emergent'].weight, first['model']['aux.emergent.weight'])
    configs = [read_json(directory / (stage + '_config.json'))
               for stage in ['pretrain_round1', 'pretrain_round2', 'finetune']]
    assert all(model_options(c) == model_options(configs[0]) for c in configs)
    codebook_config = read_json(directory / 'codebook_round2_config.json')
    assert codebook_config['runtime']['gpus'] == ''  # CPU-only offline builder, no torchrun.
    statuses = [json.loads(line) for line in (output / 'pipeline.log').read_text().splitlines() if line.startswith('{')]
    assert next(s for s in statuses if s['stage'] == 'codebook_round2_running')['processes'] == 1
    basename='shared_base_p8.json5' if args.patch_size==8 else 'shared_base.json5'
    supervised = load_config(ROOT / 'code/configs/supervised' / basename)
    formal_pretrain = load_config(ROOT / 'code/configs/pretrain' / basename)
    formal_fine = load_config(ROOT / 'code/configs/finetune' / basename)
    assert model_options(supervised) == model_options(formal_pretrain) == model_options(formal_fine)
    assert formal_pretrain['training']['epochs'] == 30
    assert Path(formal_pretrain['run']['codebook']).name == 'codebook_v1'
    assert model_options(supervised)['patch_size']==args.patch_size
    # The supervised entry also executes with the same P8 backbone.
    if args.patch_size==8:
        supervised_config=read_json(output/'finetune.json')
        supervised_config['run'].update(init_checkpoint='',warmstart_checkpoint='',output=str(output/'supervised'))
        write_json(output/'supervised.json',supervised_config)
        run('supervised',['code/training/train.py','--config',str(output/'supervised.json')])

    run('inference', ['code/infer.py', '--weights', str(directory / 'exported_model'),
        '--input', 'artifacts/reception_exploration/summary_20261114/integration_input.parquet',
        '--output', str(output / 'predictions.jsonl'), '--device', 'cpu'])
    assert len((output / 'predictions.jsonl').read_text().splitlines()) == 12
    # A completed pipeline can be resumed without retraining or rebuilding.
    before = deepcopy(stages)
    run('resume_completed', ['code/training/' + entry, '--resume', str(directory)])
    assert read_json(directory / 'manifest.json')['stages'] == before
    write_json(output / 'verification.json', {'passed': True, 'pipeline': str(directory), 'training_world_size': args.processes, 'patch_size':args.patch_size,
        'round1_steps': 3, 'round2_steps': 3, 'finetune_steps': 3, 'inference_rows': 12,
        'checks': ['real teacher vocabulary regeneration', 'backbone-only transfer with fresh structure heads',
                   'same three-stage model definition', 'exact checkpoint binding', 'completed-stage resume',
                   'allowed communication groups', 'offline vocabulary uses one process',
                   'unified export and inference'], 'meaning': 'CPU execution/transfer only; not NCCL or accuracy evidence'})
    write_json(ROOT / 'artifacts/iterative_pipeline_checks/latest.json', {'directory': str(output)})
    print('ITERATIVE PIPELINE CHECKS PASSED:', output, flush=True)


if __name__ == '__main__':
    main()

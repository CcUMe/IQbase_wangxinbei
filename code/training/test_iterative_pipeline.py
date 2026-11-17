"""Unified train.py: two-round config, independent fine config, export and inference.

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
from utils.training_workflow import read_settings, stamp, validate_devices, validate_vocabulary
from utils.run_directory import latest_run
from utils.command_config import checkpoint_reference
from training.prepare_patch_view import prepare_view
from utils.common import model_bundle, read_json, write_json
from utils.config import load_config, model_options
from utils.initialization import load_pretrained_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processes', type=int, choices=[1, 2], default=1)
    parser.add_argument('--patch-size', type=int, choices=[8, 16], default=16)
    parser.add_argument('--dictionary-metric',choices=['spherical','euclidean'],default='spherical')
    parser.add_argument('--min-frequency',type=int,default=1)
    parser.add_argument('--mask-strategy',choices=['random','bernoulli_starts'],default='random')
    parser.add_argument('--pretraining-reduction',choices=['per_record','masked_token_mean'],default='per_record')
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
        '--cluster-iterations', '2', '--batch-size', '16', '--threads', '2',
        '--metric',args.dictionary_metric,'--min-frequency',str(args.min_frequency),'--bpe-tokens','512'])
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
        config['runtime'].update(device='cpu', gpus='', processes=args.processes, precision='fp32', threads=2, log_every=1)
        config['checkpoint']['save_every_steps'] = 1
        config['evaluation']['batch_size'] = 16
        config['loss']['pretraining']['reduction']=args.pretraining_reduction
        if args.mask_strategy=='bernoulli_starts':
            config['loss']['pretraining']['mask'].update(strategy='bernoulli_starts',ratio=0.,start_probability=.06,span=8)
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
    settings = read_settings(ROOT / 'code/configs/pretrain/two_rounds_p8.json5')
    settings['configs'] = {'pretrain':str(output/'pretrain.json')}
    settings['data']['batch_size_per_gpu']=4
    settings['runtime'].update(processes=args.processes,gpu=','.join(str(i) for i in range(args.processes)))
    settings['workflow']='two_round_pretrain'
    settings['run']['output'] = str(output / 'pipeline')
    settings['round1'].update(epochs=1, codebook=str(first_book))
    settings['round2']['epochs'] = 1
    settings['codebook'].update(atomic_vocab=8, merges=4, max_patches=128,
                                cluster_iterations=2, fit_sequences=32, batch_size=16, threads=2,
                                metric=args.dictionary_metric,min_frequency=args.min_frequency,bpe_tokens=512)
    settings['runtime']['device'] = 'cpu'
    write_json(output / 'pipeline.json', settings)
    run('pipeline', ['code/training/train.py', '--config', str(output / 'pipeline.json')])
    directory = Path(read_json(output / 'pipeline/latest_pipeline.json')['pipeline_directory'])
    assert (directory / 'completion.json').is_file()
    manifest = read_json(directory / 'manifest.json')
    assert manifest['workflow']=='two_round_pretrain'
    assert set(manifest['stages'])=={'pretrain_round1','codebook_round2','pretrain_round2'}
    assert not (directory/'finetune_config.json').exists() and not (directory/'finetune').exists()
    assert not (directory/'exported_model').exists()
    pretrained=checkpoint_reference(output/'pipeline')
    assert pretrained==Path(manifest['stages']['pretrain_round2']['best_checkpoint'])
    fine=read_json(output/'finetune.json')
    fine['run'].update(init_checkpoint=str(output/'pipeline'),output=str(output/'finetuning'))
    write_json(output/'finetune.json',fine)
    run('finetune',['code/training/train.py','--config',str(output/'finetune.json')])
    tuned=latest_run(output/'finetuning')
    stages=deepcopy(manifest['stages'])
    stages['finetune']={'complete':True,'global_step':3,'run_directory':str(tuned),
                         'initialization':str(pretrained),'best_checkpoint':str(tuned/'best.pt')}
    run('export',['code/training/export.py','--checkpoint',str(tuned/'best.pt'),
                  '--output',str(output/'exported_model')])
    assert manifest['processes'] == args.processes
    for name in ['pretrain_round1', 'pretrain_round2', 'finetune']:
        assert stages[name]['complete'] and stages[name]['global_step'] == 3
        saved = torch.load(Path(stages[name]['run_directory']) / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['world_size'] == args.processes
        assert saved['training_config']['runtime']['processes'] == args.processes
    teacher = stages['pretrain_round1']['best_checkpoint']
    validate_vocabulary(stages['codebook_round2']['output'], data, teacher)
    for book in [first_book,Path(stages['codebook_round2']['output'])]:
        metadata=read_json(book/'codebook.json')
        assert metadata['metric']==args.dictionary_metric and metadata['min_frequency']==args.min_frequency
        assert metadata['fitted_training_tokens']<=512
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
               for stage in ['pretrain_round1', 'pretrain_round2']]+[read_json(output/'finetune.json')]
    assert all(model_options(c) == model_options(configs[0]) for c in configs)
    codebook_config = read_json(directory / 'codebook_round2_config.json')
    assert codebook_config['runtime']['gpus'] == ''  # CPU-only offline builder, no torchrun.
    statuses = [json.loads(line) for line in (output / 'pipeline.log').read_text().splitlines() if line.startswith('{"stage":')]
    assert next(s for s in statuses if s['stage'] == 'codebook_round2_running')['processes'] == 1
    saved_fine=torch.load(tuned/'last.pt',map_location='cpu',weights_only=True)
    assert saved_fine['training_config']['run']['init_checkpoint']==str(pretrained)
    assert saved_fine['training_config']['optimizer']['weight_decay']==.08
    assert saved_fine['training_config']['finetuning']['strategy']=='full'
    assert saved_fine['training_config']['scheduler']['name']=='cosine'
    basename='shared_base_p8.json5' if args.patch_size==8 else 'shared_base.json5'
    supervised = load_config(ROOT / 'code/configs/supervised' / basename)
    formal_pretrain = load_config(ROOT / 'code/configs/pretrain' / basename)
    formal_fine = load_config(ROOT / 'code/configs/finetune' / basename)
    assert model_options(supervised) == model_options(formal_pretrain) == model_options(formal_fine)
    assert formal_pretrain['training']['epochs'] == (50 if args.patch_size==8 else 30)
    assert Path(formal_pretrain['run']['codebook']).name == 'codebook_v1'
    assert model_options(supervised)['patch_size']==args.patch_size
    # The supervised entry also executes with the same P8 backbone.
    if args.patch_size==8:
        supervised_config=read_json(output/'finetune.json')
        supervised_config['run'].update(init_checkpoint='',warmstart_checkpoint='',output=str(output/'supervised'))
        write_json(output/'supervised.json',supervised_config)
        run('supervised',['code/training/train.py','--config',str(output/'supervised.json')])

    run('inference', ['code/infer.py', '--weights', str(output / 'exported_model'),
        '--input', 'artifacts/reception_exploration/summary_20261114/integration_input.parquet',
        '--output', str(output / 'predictions.jsonl'), '--device', 'cpu'])
    assert len((output / 'predictions.jsonl').read_text().splitlines()) == 12
    # A completed pipeline can be resumed without retraining or rebuilding.
    before = deepcopy(manifest['stages'])
    run('resume_completed', ['code/training/train.py', '--config',str(output/'pipeline.json'),'--resume', str(directory)])
    assert read_json(directory / 'manifest.json')['stages'] == before
    run('resume_finetune',['code/training/train.py','--config',str(output/'finetune.json'),
                            '--resume',str(tuned/'last.pt')])
    assert torch.load(tuned/'last.pt',map_location='cpu',weights_only=True)['global_step']==3
    manifest['workflow']='full'
    write_json(directory/'manifest.json',manifest)
    (directory/'finetune_config.json').write_text('OLD FINE CONFIG MUST NEVER BE READ')
    hashes=(directory/'source_snapshot/sha256.json').read_bytes()
    run('adopt_old_workflow',['code/training/train.py','--config',str(output/'pipeline.json'),'--resume',str(directory)])
    adopted=read_json(directory/'manifest.json')
    assert adopted['workflow']=='two_round_pretrain' and adopted['stages']==before
    assert list(directory.glob('manifest_before_split_*.json'))
    assert (directory/'source_snapshot/sha256.json').read_bytes()==hashes
    assert not (directory/'finetune').exists()
    write_json(output / 'verification.json', {'passed': True, 'pipeline': str(directory), 'training_world_size': args.processes, 'patch_size':args.patch_size,
        'dictionary_metric':args.dictionary_metric,'min_frequency':args.min_frequency,'mask_strategy':args.mask_strategy,'pretraining_reduction':args.pretraining_reduction,
        'round1_steps': 3, 'round2_steps': 3, 'finetune_steps': 3, 'inference_rows': 12,
        'checks': ['real teacher vocabulary regeneration', 'backbone-only transfer with fresh structure heads',
                   'same three-stage model definition', 'exact checkpoint binding', 'completed-stage resume',
                   'allowed communication groups', 'offline vocabulary uses one process',
                   'unified train.py for all stages', 'two-round workflow never starts finetuning',
                   'completed-workflow pointer and independent fine resume','old-workflow adoption preserves frozen trainer',
                   'unified export and inference'], 'meaning': 'CPU execution/transfer only; not NCCL or accuracy evidence'})
    write_json(ROOT / 'artifacts/iterative_pipeline_checks/latest.json', {'directory': str(output)})
    print('ITERATIVE PIPELINE CHECKS PASSED:', output, flush=True)


if __name__ == '__main__':
    main()

"""Internal two-round workflow used by the unified training CLI.

One or two GPUs train; vocabulary building uses only the first selected GPU.
Only backbone weights transfer across objectives/vocabularies. Each training
stage gets a new optimizer and schedule; --resume restores the interrupted
stage's full last.pt. Completed stages and partial vocabulary attempts survive.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import torch
from utils.common import read_config_json, read_json, write_json
from utils.config import load_config, model_options, validate_config
from utils.run_directory import latest_run

CODEBOOK_METHOD_DEFAULTS={'metric':'spherical','normalization':'rms','amplitude_percentile':99.,
    'cluster_batch_size':16384,'cluster_initializations':5,'bpe_tokens':0,'min_frequency':1}
FIELDS = {
    'configs': {'pretrain', 'finetune'}, 'run': {'output'},
    'round1': {'epochs', 'codebook'},
    'round2': {'epochs', 'learning_rate', 'backbone_lr_factor'},
    'codebook': {'atomic_vocab', 'max_patches', 'cluster_iterations',
                 'cluster_chunk_size', 'merges', 'max_span', 'fit_sequences',
                 'batch_size', 'seed', 'threads',*CODEBOOK_METHOD_DEFAULTS},
    'runtime': {'gpu', 'device', 'processes'},
    'data': {'batch_size_per_gpu'},
}


class PipelineStop(BaseException):
    def __init__(self, signum):
        self.signum = signum


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')


def absolute(value):
    return str((ROOT / Path(value).expanduser()).resolve())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_settings(path):
    settings = read_config_json(path)
    workflow=settings.pop('workflow','two_round_pretrain')
    if workflow!='two_round_pretrain':raise ValueError('Unknown workflow: '+str(workflow))
    # Backward compatible with original single-GPU pipeline configurations.
    if isinstance(settings.get('runtime'), dict):
        settings['runtime'].setdefault('processes', 1)
    settings.setdefault('data', {'batch_size_per_gpu': 0})
    if isinstance(settings.get('codebook'),dict):
        for key,value in CODEBOOK_METHOD_DEFAULTS.items():settings['codebook'].setdefault(key,value)
    if set(settings) != set(FIELDS):
        raise ValueError('Pipeline config groups must be: ' + ', '.join(FIELDS))
    for group, names in FIELDS.items():
        if group=='configs' and isinstance(settings[group],dict):
            names={'pretrain'} if 'finetune' not in settings[group] else names
        if not isinstance(settings[group], dict) or set(settings[group]) != names:
            raise ValueError(f'Invalid {group} fields; expected {sorted(names)}')
    for group in ['round1', 'round2']:
        if type(settings[group]['epochs']) is not int or settings[group]['epochs'] < 1:
            raise ValueError(group + '.epochs must be a positive integer')
    for name in ['learning_rate', 'backbone_lr_factor']:
        value = settings['round2'][name]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('round2.' + name + ' must be positive and finite')
    for name, value in settings['codebook'].items():
        if name in ['metric','normalization','amplitude_percentile']:continue
        if type(value) is not int or value < (0 if name in ['merges', 'seed','bpe_tokens'] else 1):
            raise ValueError('Invalid codebook.' + name)
    book = settings['codebook']
    if book['metric'] not in ['spherical','euclidean'] or book['normalization'] not in ['rms','amplitude_percentile'] or not 0<book['amplitude_percentile']<=100:
        raise ValueError('Invalid dictionary metric/normalization')
    if book['atomic_vocab'] < 2 or book['atomic_vocab'] + book['merges'] >= 65535:
        raise ValueError('Invalid structural vocabulary size')
    batch = settings['data']['batch_size_per_gpu']
    if type(batch) is not int or batch < 0:
        raise ValueError('data.batch_size_per_gpu must be nonnegative; 0 keeps the stage template batch')
    validate_devices(settings['runtime']['gpu'], settings['runtime']['processes'], settings['runtime']['device'])
    return settings


def validate_devices(gpu, processes, device):
    if type(processes) is not int or processes not in [1, 2] or device not in ['auto', 'cpu']:
        raise ValueError('Choose one or two training processes, device auto/cpu')
    if not isinstance(gpu, str):
        raise ValueError('GPU selection must be a string')
    cards = [part.strip() for part in gpu.split(',')]
    if len(cards) != processes or any(not card.isdigit() for card in cards) or len(set(cards)) != processes:
        raise ValueError('GPU indices must be distinct and match the number of training processes')
    if processes == 2 and set(cards) not in [{'0', '1'}, {'2', '3'}, {'4', '5'}]:
        raise ValueError('Dual GPU requires communication group 0,1 or 2,3 or 4,5')
    return cards


def validate_vocabulary(path, data, teacher=None):
    """Check metadata, label alignment and completed artifacts, not just paths."""
    path = Path(path)
    info = read_json(Path(data) / 'dataset.json')
    book = read_json(path / 'codebook.json')
    if book['dataset_fingerprint'] != info['fingerprint'] or book['patch_size'] != info['patch_size']:
        raise ValueError('Vocabulary/data fingerprint or patch size mismatch')
    if teacher is None:
        if book['feature_kind'] != 'raw_iq' or book.get('teacher_checkpoint'):
            raise ValueError('Round1 must use a raw-IQ vocabulary, without any old teacher')
    elif (book['feature_kind'] != 'encoder' or not book.get('teacher_checkpoint')
          or Path(book['teacher_checkpoint']).resolve() != Path(teacher).resolve()):
        raise ValueError('Round2 vocabulary must come from THIS pipeline round1 model')
    if not (path / 'codebook.pt').is_file():
        raise FileNotFoundError(path / 'codebook.pt')
    lengths = np.load(Path(data) / 'metadata.npz')['length']
    counts = lengths // book['patch_size']
    columns = np.arange(1024 // book['patch_size'])[None]
    for kind in ['atomic', 'emergent']:
        labels = np.load(path / (kind + '_labels.npy'), mmap_mode='r')
        if labels.shape != (info['rows'], columns.shape[1]):
            raise ValueError(kind + ' label shape mismatch')
        for start in range(0, len(labels), 8192):
            part = labels[start:start + 8192]
            valid = columns < counts[start:start + 8192, None]
            if np.any(part[valid] >= book[kind + '_vocab']) or np.any(part[~valid] != 65535):
                raise ValueError(kind + ' has invalid or misaligned structure labels')
    return book


def prepare_stage_config(path,gpu,device):
    c=load_config(path)
    if c['training']['stage']=='pretrain':
        aug=c['data']['augmentation']
        if aug['phase_rotation_probability'] or any(aug['modulation_only'][k] for k in ['phase_rotation_probability','circular_shift_probability']):
            raise ValueError('Pretraining structure targets require unaugmented IQ')
    if c['training']['max_steps']:raise ValueError('Use complete epoch budgets in the automatic pipeline')
    if c['checkpoint']['resume_from']:raise ValueError('Use pipeline --resume, not a config resume_from')
    if not c['checkpoint']['save_last'] or not c['checkpoint']['save_best']:raise ValueError('Pipeline requires last/best checkpoint saving')
    if c['evaluation']['early_stopping']['patience']:raise ValueError('This full-budget pipeline requires early stopping disabled')
    c['runtime'].update(device=device,gpus=gpu if device!='cpu' else '',processes=1)
    for key in ['data','codebook','init_checkpoint','warmstart_checkpoint']:
        if c['run'][key]:c['run'][key]=absolute(c['run'][key])
    return c



def initial_configs(settings, gpu, device, processes=1):
    cards = validate_devices(gpu, processes, device)
    first=prepare_stage_config(absolute(settings['configs']['pretrain']),cards[0],device)
    if first['training']['stage']!='pretrain':raise ValueError('Expected a structural pretraining config')
    if first['run']['init_checkpoint'] or first['run']['warmstart_checkpoint']:
        raise ValueError('Round1 must initialize the current architecture from scratch')
    first['run']['codebook'] = absolute(settings['round1']['codebook'])
    first['training']['epochs'] = settings['round1']['epochs']
    vocabulary=validate_vocabulary(first['run']['codebook'], first['run']['data'])
    for key,default in [('metric','spherical'),('normalization','rms'),('amplitude_percentile',99.),('min_frequency',1)]:
        actual=vocabulary.get(key,default)
        requested=settings['codebook'].get(key,default)
        if actual!=requested:
            raise ValueError(f'Round1 vocabulary differs from the requested method: {key}; '
                f'vocabulary={actual!r}, config codebook.{key}={requested!r}; '
                f'vocabulary directory={first["run"]["codebook"]}. '
                'Align the workflow codebook settings with this vocabulary, or build a new vocabulary; '
                'changing only round1.codebook does not change the round2 construction method.')
    if vocabulary.get('normalization','rms')!=first['model']['signal']['normalization']:
        raise ValueError('Dictionary and backbone normalization must match')
    second = deepcopy(first)
    second['training']['epochs'] = settings['round2']['epochs']
    second['experiment'].update(name='pretrain_shared_base_round2',
        description='Round1 current-backbone initialization; regenerated new-teacher structural targets')
    second['optimizer'].update(learning_rate=settings['round2']['learning_rate'],
        backbone_lr_factor=settings['round2']['backbone_lr_factor'])
    # Deliberately unresolved until this run's teacher/vocabulary are complete.
    second['run'].update(codebook='', init_checkpoint='', warmstart_checkpoint='')
    configs={'pretrain_round1':first,'pretrain_round2':second}
    for config in configs.values():
        config['runtime'].update(gpus=','.join(cards) if device != 'cpu' else '', processes=processes)
        batch = settings.get('data', {}).get('batch_size_per_gpu', 0)
        if batch:
            config['data']['batch_size'] = batch
        validate_config(config)
    return configs


class Pipeline:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = directory, manifest
        self.child = None
        self.lock = (directory / 'pipeline.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
        self.env = {**os.environ, 'CUDA_VISIBLE_DEVICES': manifest['gpu'] if manifest['device'] != 'cpu' else '',
                    'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
        for signum in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(signum, self.stop)

    def save(self):
        temporary = self.directory / 'manifest.tmp'
        write_json(temporary, self.manifest)
        temporary.replace(self.directory / 'manifest.json')

    def status(self, stage, **extra):
        value = {'stage': stage, 'pipeline_pid': os.getpid(),
                 'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra}
        temporary = self.directory / 'status.tmp'
        write_json(temporary, value)
        temporary.replace(self.directory / 'status.json')
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def stop(self, signum, _):
        for item in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(item, signal.SIG_IGN)
        if self.child is not None and self.child.poll() is None:
            # The child owns a session: stop torchrun AND its worker processes.
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Unwind Popen.wait before waiting again: its waitpid lock can be held
        # when a Python signal handler interrupts the original blocking wait.
        raise PipelineStop(signum)

    def finish_stop(self, signum):
        if self.child is not None:
            try:
                self.child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child.wait(timeout=10)
        self.status('interrupted', signal=signum)
        raise SystemExit(128 + signum)

    def execute(self, stage, script, args, cpu=False, single_gpu=False):
        console = self.directory / (stage + '_console.log')
        command = [sys.executable, '-u', str(Path(self.manifest['code_snapshot']) / 'training' / script), *args]
        env = dict(self.env)
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        elif single_gpu and self.manifest['device'] != 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = self.manifest['gpu'].split(',')[0]
        stream_console=self.manifest.get('stream_console',False)
        with console.open('ab', buffering=0) as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE if stream_console else log,
                                          stderr=subprocess.STDOUT,env=env,start_new_session=True,bufsize=0)
            processes = 1 if cpu or single_gpu else self.manifest.get('processes', 1)
            self.status(stage + '_running', child_pid=self.child.pid, console=str(console), processes=processes)
            if stream_console:
                while chunk:=self.child.stdout.read(8192):
                    log.write(chunk)
                    if hasattr(sys.stdout,'buffer'):sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
                    else:sys.stdout.write(chunk.decode(errors='replace'));sys.stdout.flush()
            result = self.child.wait()
            self.child = None
        if result:
            raise RuntimeError(f'{stage} exited {result}; inspect {console}')

    def train(self, stage, config):
        complete = self.manifest['stages'].get(stage, {})
        if complete.get('complete'):
            if not Path(complete['best_checkpoint']).is_file():
                raise FileNotFoundError(complete['best_checkpoint'])
            return complete['best_checkpoint']
        config_path = self.directory / (stage + '_config.json')
        write_json(config_path, config)
        args = ['--config', str(config_path)]
        try:
            run = latest_run(config['run']['output'])
        except FileNotFoundError:
            run = None
        if run is not None and (run / 'last.pt').is_file():
            args += ['--resume', str(run / 'last.pt')]
        self.execute(stage, 'train.py', args)
        run = latest_run(config['run']['output'])
        last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True, mmap=True)
        if not last['epoch_complete'] or last['epoch'] + 1 < config['training']['epochs']:
            raise RuntimeError(stage + ' did not finish its configured epoch budget')
        best = run / 'best.pt'
        if not best.is_file():
            raise FileNotFoundError(best)
        self.manifest['stages'][stage] = {'complete': True, 'run_directory': str(run),
            'best_checkpoint': str(best), 'global_step': last['global_step'],
            'epochs': config['training']['epochs'], 'initialization': config['run']['init_checkpoint'],
            'codebook': config['run']['codebook']}
        del last
        self.save()
        self.status(stage + '_complete', checkpoint=str(best))
        return str(best)

    def build_vocabulary(self, teacher, data):
        stage = 'codebook_round2'
        previous = self.manifest['stages'].get(stage, {})
        if previous.get('output'):
            candidate = Path(previous['output'])
            if (candidate / 'codebook.json').is_file():
                # codebook.json is written LAST by the builder: completed even
                # if the parent was interrupted before updating its manifest.
                validate_vocabulary(candidate, data, teacher)
                previous['complete'] = True
                self.save()
                return str(candidate)
        # Never overwrite an interrupted attempt or either historical vocabulary.
        output = self.directory / 'codebooks' / ('round2_' + stamp())
        book = self.manifest['settings']['codebook']
        grouped = {'run': {'data': data, 'output': str(output), 'teacher_checkpoint': teacher},
            'atomic': {'vocab_size': book['atomic_vocab'], 'max_patches': book['max_patches'],
                       'iterations': book['cluster_iterations'], 'chunk_size': book['cluster_chunk_size'],
                       'metric':book.get('metric','spherical'),'normalization':book.get('normalization','rms'),
                       'amplitude_percentile':book.get('amplitude_percentile',99.),
                       'batch_size':book.get('cluster_batch_size',16384),'initializations':book.get('cluster_initializations',5)},
            'emergent': {'merges': book['merges'], 'max_span': book['max_span'], 'fit_sequences': book['fit_sequences'],
                         'fit_tokens':book.get('bpe_tokens',0),'min_frequency':book.get('min_frequency',1)},
            'runtime': {'batch_size': book['batch_size'], 'seed': book['seed'], 'threads': book['threads'],
                        'device': self.manifest['device'], 'gpus': self.manifest['gpu'].split(',')[0] if self.manifest['device'] != 'cpu' else ''}}
        config_path = self.directory / 'codebook_round2_config.json'
        write_json(config_path, grouped)
        attempts = previous.get('attempts', []) + [str(output)]
        self.manifest['stages'][stage] = {'complete': False, 'output': str(output),
            'teacher_checkpoint': teacher, 'teacher_sha256': digest(teacher), 'attempts': attempts}
        self.save()
        self.execute(stage, 'build_codebook.py', ['--config', str(config_path)], single_gpu=True)
        vocabulary = validate_vocabulary(output, data, teacher)
        self.manifest['stages'][stage].update(complete=True, atomic_vocab=vocabulary['atomic_vocab'],
                                            emergent_vocab=vocabulary['emergent_vocab'])
        self.save()
        self.status(stage + '_complete', output=str(output), teacher=teacher)
        return str(output)

    def run(self):
        configs = {name: read_json(self.directory / (name + '_config.json'))
                   for name in ['pretrain_round1', 'pretrain_round2']}
        teacher = self.train('pretrain_round1', configs['pretrain_round1'])
        vocabulary = self.build_vocabulary(teacher, configs['pretrain_round1']['run']['data'])
        second = configs['pretrain_round2']
        second['run'].update(init_checkpoint=teacher, codebook=vocabulary, warmstart_checkpoint='')
        pretrained = self.train('pretrain_round2', second)
        result={'status':'pretrain_complete','workflow':'two_round_pretrain','pipeline_directory':str(self.directory),
                'checkpoint':pretrained,'stages':self.manifest['stages']}
        write_json(self.directory/'pretraining_completion.json',result)
        write_json(self.directory/'completion.json',result)
        write_json(self.directory.parent/'latest_pretrained.json',result)
        self.status('pretrain_complete',checkpoint=pretrained)



def run_workflow():
    """Called only by train.py after examining the configuration type."""
    description='Two-round structural pretraining; stops without finetuning. All stages use the shared trainer.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu', '--gpus', dest='gpu', help='Physical GPU index or supported pair')
    parser.add_argument('--processes', type=int, choices=[1, 2], help='Override training process count')
    parser.add_argument('--device', choices=['auto', 'cpu'])
    parser.add_argument('--resume', type=Path, help='Continue a frozen pipeline directory')
    parser.add_argument('--batch-size',type=int,help='Override per-process training batch')
    parser.add_argument('--output',type=Path,help='Override workflow output root')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or 'LOCAL_RANK' in os.environ:
        parser.error('Run one pipeline coordinator directly; it launches distributed training itself')
    if args.resume:
        if args.batch_size is not None or args.output is not None:parser.error('Resume keeps the frozen batch and output')
        directory = args.resume.resolve()
        manifest = read_json(directory / 'manifest.json')
        gpu, device = manifest['gpu'], manifest['device']
        processes = manifest.get('processes', 1)
        if manifest.get('workflow')!='two_round_pretrain':
            if manifest['stages'].get('finetune') or (directory/'finetune/latest_run.json').exists():
                raise ValueError('This workflow has already started finetuning; cannot convert it to pretrain-only')
        if ((args.gpu is not None and args.gpu != gpu) or (args.device is not None and args.device != device)
                or (args.processes is not None and args.processes != processes)):
            parser.error('Resume with the original GPU/device/process count')
        snapshot = Path(manifest['code_snapshot'])
        for relative, expected in read_json(directory / 'source_snapshot/sha256.json').items():
            if digest(snapshot / relative) != expected:
                raise ValueError('Frozen source changed: ' + relative)
    else:
        settings = read_settings(args.config)
        gpu = args.gpu if args.gpu is not None else settings['runtime']['gpu']
        device = args.device if args.device is not None else settings['runtime']['device']
        processes = args.processes if args.processes is not None else settings['runtime']['processes']
    cards = validate_devices(gpu, processes, device)
    gpu = ','.join(cards)
    if not args.resume:
        if args.batch_size is not None:settings['data']['batch_size_per_gpu']=args.batch_size
        if args.output is not None:settings['run']['output']=str(args.output.resolve())
        configs = initial_configs(settings, gpu, device, processes)
        directory = Path(absolute(settings['run']['output'])) / stamp()
        directory.mkdir(parents=True, exist_ok=False)
        settings['runtime'].update(gpu=gpu, device=device, processes=processes)
        write_json(directory / 'pipeline_config.json', settings)
        for stage, config in configs.items():
            config['run']['output'] = str(directory / stage)
            write_json(directory / (stage + '_config.json'), config)
        snapshot = directory / 'source_snapshot/code'
        shutil.copytree(ROOT / 'code', snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        write_json(directory / 'source_snapshot/sha256.json',
                   {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*') if p.is_file()})
        manifest = {'format_version': 3, 'gpu': gpu, 'device': device, 'processes': processes, 'settings': settings,
                    'code_snapshot': str(snapshot), 'stages': {},
                    'workflow':'two_round_pretrain',
                    'architecture_sha256': hashlib.sha256(json.dumps(model_options(configs['pretrain_round1']), sort_keys=True).encode()).hexdigest()}
    pipeline = Pipeline(directory, manifest)
    if manifest.get('workflow')!='two_round_pretrain':
        # Conversion changes orchestration only. The trainer's frozen source,
        # exact stage configs, vocabulary, optimizer and schedule are preserved.
        write_json(directory/('manifest_before_split_'+stamp()+'.json'),deepcopy(manifest))
        manifest.setdefault('workflow_history',[]).append({'from':manifest.get('workflow','full'),
            'to':'two_round_pretrain','time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
        manifest['workflow']='two_round_pretrain'
    manifest['stream_console']=True
    pipeline.save()
    write_json(directory.parent / 'latest_pipeline.json', {'pipeline_directory': str(directory), 'pid': os.getpid()})
    pipeline.status('starting', directory=str(directory), gpu=gpu)
    try:
        pipeline.run()
    except PipelineStop as stop:
        pipeline.finish_stop(stop.signum)
    except Exception as error:
        pipeline.status('failed', error=repr(error))
        raise


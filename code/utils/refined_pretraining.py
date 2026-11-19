"""Frozen teacher -> audited vocabulary -> independent second-round training."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sys
from utils.common import read_config_json,read_json,write_json
from utils.config import load_config,validate_config
from utils.command_config import checkpoint_reference
from utils.training_workflow import ROOT,Pipeline,PipelineStop,absolute,digest,stamp,validate_devices
from utils.vocabulary_audit import audit_vocabulary


def run_refined():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or 'LOCAL_RANK' in os.environ:p.error('Start one coordinator')
    if args.resume:
        directory=args.resume.resolve();manifest=read_json(directory/'manifest.json')
        if manifest.get('workflow')!='refine_pretrain':raise ValueError('Wrong workflow')
        for name,expected in read_json(directory/'source_snapshot/sha256.json').items():
            if digest(Path(manifest['code_snapshot'])/name)!=expected:raise ValueError('Frozen source changed: '+name)
        if digest(manifest['teacher_checkpoint'])!=manifest['teacher_sha256']:raise ValueError('Frozen teacher changed')
        if digest(directory/'requested_round2_config.json')!=manifest['training_config_sha256']:raise ValueError('Frozen training config changed')
    else:
        settings=read_config_json(args.config)
        if set(settings)!={'workflow','run','configs','runtime','codebook'} or settings['workflow']!='refine_pretrain':raise ValueError('Invalid refined pretraining settings')
        runtime=settings['runtime'];cards=validate_devices(runtime['gpus'],runtime['processes'],runtime['device'])
        teacher=checkpoint_reference(Path(absolute(settings['run']['teacher_checkpoint'])))
        config=load_config(absolute(settings['configs']['pretrain']))
        if config['training']['stage']!='pretrain' or config['run']['warmstart_checkpoint'] or config['checkpoint']['resume_from']:
            raise ValueError('Expected fresh second-round backbone initialization')
        if config['run']['init_checkpoint'] and checkpoint_reference(Path(absolute(config['run']['init_checkpoint'])))!=teacher:
            raise ValueError('Training and vocabulary must bind to the same teacher')
        directory=Path(absolute(settings['run']['output']))/stamp();directory.mkdir(parents=True,exist_ok=False)
        snapshot=directory/'source_snapshot/code'
        shutil.copytree(ROOT/'code',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write_json(directory/'source_snapshot/sha256.json',{str(f.relative_to(snapshot)):digest(f) for f in snapshot.rglob('*') if f.is_file()})
        config['run'].update(data=absolute(config['run']['data']),output=str(directory/'pretrain_round2'),init_checkpoint=str(teacher),codebook='')
        config['runtime'].update(gpus=','.join(cards),processes=runtime['processes'],device=runtime['device'])
        validate_config(config);write_json(directory/'requested_round2_config.json',config)
        write_json(directory/'refine_config.json',settings)
        manifest={'workflow':'refine_pretrain','format_version':1,'gpu':','.join(cards),'device':runtime['device'],
                  'processes':runtime['processes'],'code_snapshot':str(snapshot),'stages':{},'settings':settings,
                  'teacher_checkpoint':str(teacher),'teacher_sha256':digest(teacher),'stream_console':True,
                  'training_config_sha256':digest(directory/'requested_round2_config.json')}
    pipeline=Pipeline(directory,manifest);pipeline.save()
    write_json(directory.parent/'latest_pipeline.json',{'pipeline_directory':str(directory),'pid':os.getpid()})
    try:
        config=read_json(directory/'requested_round2_config.json')
        vocabulary=pipeline.build_vocabulary(manifest['teacher_checkpoint'],config['run']['data'])
        pipeline.status('vocabulary_audit_running',vocabulary=vocabulary)
        # Call the frozen audit module in a fresh CPU child, never the live repo implementation.
        pipeline.execute('vocabulary_audit','audit_structure_vocabulary.py',
                         ['--data',config['run']['data'],'--codebook',vocabulary],cpu=True)
        config['run']['codebook']=vocabulary
        checkpoint=pipeline.train('pretrain_round2',config)
        result={'status':'pretrain_complete','workflow':'refine_pretrain','checkpoint':checkpoint,'pipeline_directory':str(directory)}
        write_json(directory/'completion.json',result);write_json(directory/'pretraining_completion.json',result)
        write_json(directory.parent/'latest_pretrained.json',result);pipeline.status('pretrain_complete',checkpoint=checkpoint)
    except PipelineStop as stop:pipeline.finish_stop(stop.signum)
    except Exception as error:
        pipeline.status('failed',error=repr(error));raise

"""Config-only CLI paths, legacy CLI overrides, and safe automatic launching."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.command_config import command_defaults, configured_launch, checkpoint_reference


class CommandConfigTests(unittest.TestCase):
    def test_config_satisfies_required_paths_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'run.json'
            config.write_text(json.dumps({'run':{'data':'cache','teacher':''}}))
            parser=argparse.ArgumentParser()
            parser.add_argument('--config')
            parser.add_argument('--data',type=Path,required=True)
            parser.add_argument('--teacher',type=Path)
            argv=['--config',str(config)]
            command_defaults(parser,{'run':{'data':'data','teacher':'teacher'}},argv)
            args=parser.parse_args(argv)
            self.assertEqual(args.data,Path('cache'))
            self.assertIsNone(args.teacher)
            self.assertEqual(parser.parse_args([*argv,'--data','other']).data,Path('other'))

    def test_directory_reference_resolves_latest_best(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);run=parent/'stamp';run.mkdir();best=run/'best.pt';best.touch()
            (parent/'latest_run.json').write_text(json.dumps({'run_directory':str(run)}))
            self.assertEqual(checkpoint_reference(parent),best)
            self.assertEqual(checkpoint_reference(run),best)
            self.assertEqual(checkpoint_reference(best),best)
            with self.assertRaises((ValueError,FileNotFoundError)):
                checkpoint_reference(parent/'missing.pt')

    def test_auto_ddp_uses_current_python_and_does_not_recurse(self):
        with patch.dict(os.environ,{},clear=True), patch.object(sys,'argv',['train.py','--config','config.json']), \
             patch('utils.command_config.os.execvpe', side_effect=SystemExit(0)) as run:
            with self.assertRaises(SystemExit) as exc:
                configured_launch('2,3',2,distributed=True)
            self.assertEqual(exc.exception.code,0)
            self.assertEqual(run.call_args.args[0],sys.executable)
            self.assertEqual(run.call_args.args[1][:3],[sys.executable,'-m','torch.distributed.run'])
            self.assertEqual(run.call_args.args[2]['CUDA_VISIBLE_DEVICES'],'2,3')
            self.assertEqual(run.call_args.args[2]['TORCH_ELASTIC_SHUTDOWN_TIMEOUT'],'10')
        with patch.dict(os.environ,{'LOCAL_RANK':'0','WORLD_SIZE':'2','CUDA_VISIBLE_DEVICES':'0,1'}), \
             patch('utils.command_config.os.execvpe') as run:
            configured_launch('2,3',2,distributed=True)
            run.assert_not_called()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0,1')

    def test_completed_round2_pointer_and_pending_round2(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pipeline=root/'pipeline';pipeline.mkdir()
            first=pipeline/'round1.pt';first.touch()
            second=pipeline/'round2.pt';second.touch()
            manifest={'stages':{'pretrain_round1':{'complete':True,'best_checkpoint':str(first)},
                                'pretrain_round2':{'complete':False}}}
            (pipeline/'manifest.json').write_text(json.dumps(manifest))
            (root/'latest_pipeline.json').write_text(json.dumps({'pipeline_directory':str(pipeline)}))
            with self.assertRaisesRegex(ValueError,'pretrain_round2'):checkpoint_reference(root)
            self.assertEqual(checkpoint_reference(first),first)
            manifest['stages']['pretrain_round2']={'complete':True,'best_checkpoint':str(second)}
            (pipeline/'manifest.json').write_text(json.dumps(manifest))
            pointer={'pipeline_directory':str(pipeline),'checkpoint':str(second)}
            (root/'latest_pretrained.json').write_text(json.dumps(pointer))
            self.assertEqual(checkpoint_reference(root),second)
            self.assertEqual(checkpoint_reference(pipeline),second)
            pointer['checkpoint']=str(first)
            (root/'latest_pretrained.json').write_text(json.dumps(pointer))
            with self.assertRaisesRegex(ValueError,'does not match'):checkpoint_reference(root)

    def test_checkpoint_pointer_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'latest_pipeline.json').write_text(json.dumps({'pipeline_directory':str(root)}))
            with self.assertRaisesRegex(ValueError,'Cyclic'):checkpoint_reference(root)

    def test_launch_preserves_explicit_shutdown_timeout(self):
        with patch.dict(os.environ, {'TORCH_ELASTIC_SHUTDOWN_TIMEOUT': '7'}, clear=True), \
             patch('utils.command_config.os.execvpe', side_effect=SystemExit(0)) as launch:
            with self.assertRaises(SystemExit):
                configured_launch('2,3', 2, distributed=True)
            self.assertEqual(launch.call_args.args[2]['TORCH_ELASTIC_SHUTDOWN_TIMEOUT'], '7')

    def test_invalid_pair_is_rejected_before_process_launch(self):
        with patch.dict(os.environ,{},clear=True),patch('utils.command_config.os.execvpe') as run:
            with self.assertRaisesRegex(ValueError,'Two-GPU'):
                configured_launch('2,4',2,distributed=True)
            with self.assertRaisesRegex(ValueError,'one process'):
                configured_launch('2,3',2,distributed=False)
            run.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)

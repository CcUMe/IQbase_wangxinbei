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
             patch('utils.command_config.subprocess.run') as run:
            run.return_value.returncode=0
            with self.assertRaises(SystemExit) as exc:
                configured_launch('2,3',2,distributed=True)
            self.assertEqual(exc.exception.code,0)
            self.assertEqual(run.call_args.args[0][:3],[sys.executable,'-m','torch.distributed.run'])
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'2,3')
        with patch.dict(os.environ,{'LOCAL_RANK':'0','WORLD_SIZE':'2','CUDA_VISIBLE_DEVICES':'0,1'}), \
             patch('utils.command_config.subprocess.run') as run:
            configured_launch('2,3',2,distributed=True)
            run.assert_not_called()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0,1')

    def test_invalid_pair_is_rejected_before_process_launch(self):
        with patch.dict(os.environ,{},clear=True),patch('utils.command_config.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError,'Two-GPU'):
                configured_launch('2,4',2,distributed=True)
            with self.assertRaisesRegex(ValueError,'one process'):
                configured_launch('2,3',2,distributed=False)
            run.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)

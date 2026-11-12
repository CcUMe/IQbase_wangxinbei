"""Export exactly one inference model with the training-only heads removed."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.rf_model import RFModel
from utils.common import load_checkpoint, load_safetensors, model_bundle, save_safetensors, write_json
from utils.command_config import command_defaults, checkpoint_reference


def main():
    import torch
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('weights'))
    command_defaults(p, {'run': {'checkpoint': 'checkpoint', 'output': 'output'}})
    args = p.parse_args()
    args.checkpoint = checkpoint_reference(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint['stage'] != 'supervised':
        raise ValueError('Export a jointly supervised/finetuned checkpoint; pretraining heads alone are not submission-ready')
    bundle = model_bundle(checkpoint)
    bundle['model_config'] = {**bundle['model_config'], 'pretrain_heads': False, 'atomic_vocab': 0, 'emergent_vocab': 0,
                              'contrastive_head': False}
    model = RFModel(**bundle)
    state = {k: v for k, v in checkpoint['model'].items() if not k.startswith('aux.')}
    model.load_state_dict(state, strict=True)
    report = model.parameter_report()
    if report['shared_ratio'] <= .8 or report['final_parameters'] > 1_000_000_000:
        raise ValueError(f'Contest parameter constraint failed: {report}')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'model.safetensors'
    save_safetensors(path, model.state_dict())
    # Verify every exported tensor before declaring the artifact usable.
    restored = load_safetensors(path)
    if set(restored) != set(model.state_dict()) or any(not torch.equal(restored[k], v.cpu()) for k, v in model.state_dict().items()):
        raise ValueError('Export round-trip mismatch')
    write_json(args.output / 'config.json', {**bundle, 'format_version': 1, 'parameters': report,
        'source_checkpoint': str(args.checkpoint.resolve()),
        'trained_steps': checkpoint['global_step'] + checkpoint.get('repair_source_steps', checkpoint.get('initialization_source_steps', 0)),
        'validation_metrics': checkpoint.get('metrics', {}),
        'note': 'One shared model. Recorded validation metrics are local and do not equal the official competition score.'})
    print(report)


if __name__ == '__main__':
    main()

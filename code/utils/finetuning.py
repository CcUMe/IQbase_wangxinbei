"""Deterministic fine-tuning phases, frozen module modes and optional L2-SP."""
import torch


class FineTuneController:
    def __init__(self, model, settings, anchor=None):
        self.model, self.settings = model, settings
        self.phase = None
        self.anchor = None
        if settings['anchor_weight']:
            source = anchor if anchor is not None else model.backbone.state_dict()
            parameters = dict(model.backbone.named_parameters())
            if set(source) != set(parameters):
                # Only parameters are anchored; ignore registered buffers in state_dict.
                source = {k: v for k, v in source.items() if k in parameters}
            if set(source) != set(parameters):
                raise ValueError('Fine-tuning anchor does not match backbone parameters')
            self.anchor = {k: v.detach().to(parameters[k].device).clone() for k, v in source.items()}

    def phase_for_step(self, step):
        if self.settings['strategy'] == 'full':
            return 'full'
        if step < self.settings['head_warmup_steps']:
            return 'heads'
        if step < self.settings['head_warmup_steps'] + self.settings['upper_layers_steps']:
            return 'upper'
        return 'full'

    def is_boundary(self, step):
        return self.settings['strategy'] == 'gradual' and step > 0 and step in {
            self.settings['head_warmup_steps'],
            self.settings['head_warmup_steps'] + self.settings['upper_layers_steps']}

    def apply(self, step):
        phase = self.phase_for_step(step)
        changed = phase != self.phase
        if changed:
            depth = len(self.model.backbone.blocks)
            first = depth - self.settings['unfreeze_last_n_blocks']
            for name, p in self.model.named_parameters():
                allowed = not name.startswith('backbone.') or phase == 'full'
                if phase == 'upper' and name.startswith('backbone.'):
                    allowed = name.startswith('backbone.norm.')
                    if name.startswith('backbone.blocks.'):
                        allowed = int(name.split('.')[2]) >= first
                p.requires_grad_(allowed)
                if not allowed:
                    p.grad = None
            self.phase = phase
        self.train_mode()
        return changed

    def train_mode(self):
        self.model.train()
        if self.phase != 'full':
            self.model.backbone.eval()
            if self.phase == 'upper':
                for block in self.model.backbone.blocks[-self.settings['unfreeze_last_n_blocks']:]:
                    block.train()
                self.model.backbone.norm.train()

    def anchor_loss(self):
        if self.anchor is None:
            return None
        terms, count = [], 0
        for name, p in self.model.backbone.named_parameters():
            if p.requires_grad:
                terms.append((p.float() - self.anchor[name].float()).square().sum())
            count += p.numel()
        return sum(terms) / count if terms else None

    def state(self, next_step):
        return {'applied_phase': self.phase, 'next_phase': self.phase_for_step(next_step),
                'next_step': next_step,
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    def anchor_state(self):
        return {k: v.cpu() for k, v in self.anchor.items()} if self.anchor is not None else None

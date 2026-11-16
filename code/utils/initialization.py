"""Backbone-only transfer with explicit semantic and new-adapter checks."""
import torch
from models.iq_geometry import iq_geometry_options
from models.patch_shift import patch_shift_options
from models.density_encoding import density_options
from utils.position import POSITION_DEFAULTS


def load_pretrained_backbone(model,checkpoint):
    old=checkpoint['model_config'];new=model.config
    defaults=dict(width=384,depth=8,heads=6,patch_size=16,stft_window=64,
        mlp_ratio=4.,phase_features=False,position='samples',rope_base=10000.,
        qk_normalization=False,qk_scale=1.,normalize_fusion=False,rms_epsilon=1e-6,
        global_spectrum=True,scale_embedding=True,global_spectrum_input=True,
        frontend='time_frequency',normalization='rms',amplitude_percentile=99.)
    for key,default in defaults.items():
        if old.get(key,default)!=new.get(key,default):
            raise ValueError('Pretrained backbone config mismatch: '+key)
    if {**POSITION_DEFAULTS,**old.get('position_options',{})}!=new['position_options']:
        raise ValueError('Pretrained backbone position_options mismatch')
    shared={key[len('backbone.'):]:value for key,value in checkpoint['model'].items() if key.startswith('backbone.')}
    allowed=[]
    for name,normalize,prefix in [('iq_geometry',lambda x:iq_geometry_options(x,new['patch_size']),'iq_geometry.'),
                                  ('patch_shift',lambda x:patch_shift_options(x,new['patch_size']),'patch_shift.'),
                                  ('density_encoding',density_options,'time.density.')]:
        before=normalize(old.get(name));after=normalize(new.get(name))
        if before['enabled'] and before!=after:
            raise ValueError('Pretrained backbone cannot discard or reinterpret '+name)
        if after['enabled'] and not before['enabled']:
            allowed.append(prefix)
            module=getattr(model.backbone,name) if name!='density_encoding' else model.backbone.time.density
            with torch.no_grad():
                (module.shared_projection if name=='density_encoding' else module.up).weight.zero_()
    for name,prefix in [('local_refinement','time.refinement.'),('multiscale_frontend','time.multiscale.')]:
        if old.get(name,False) and not new.get(name,False):raise ValueError('Cannot discard pretrained '+name)
        if new.get(name,False) and not old.get(name,False):allowed.append(prefix)
    expected=model.backbone.state_dict()
    missing=[key for key in expected if key not in shared and not key.startswith(tuple(allowed))]
    unexpected=[key for key in shared if key not in expected]
    if missing or unexpected:raise ValueError(f'Incompatible pretrained backbone: missing={missing}, unexpected={unexpected}')
    # Heads are deliberately untouched, including random task heads for finetuning
    # and newly sized structure heads when starting another vocabulary round.
    return model.backbone.load_state_dict(shared,strict=False)

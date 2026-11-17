"""Pre-flight unit tests (guide 6.3 check 3). Run: python code/training/unit_tests.py"""
import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tfuniem import ConvSTFT, GlobalSpectrum, RepBlock, TFUniEM, count_parameters

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ok = True


def check(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f'[{"PASS" if good else "FAIL"}] {name}: max err {err:.3e} (tol {tol:.0e})')


# 1. ConvSTFT vs torch.stft
for L in (512, 1024):
    x = torch.randn(4, 2, L, dtype=torch.float64)
    m = ConvSTFT(64, 4).double()
    ours = m(x)  # log power, (B, 64, L/4), bins -32..31
    z = torch.complex(x[:, 0], x[:, 1])
    zp = torch.nn.functional.pad(torch.view_as_real(z).permute(0, 2, 1), (30, 30)).permute(0, 2, 1).contiguous()
    zp = torch.view_as_complex(zp)
    ref = torch.stft(zp, 64, 4, window=torch.hann_window(64, periodic=True, dtype=torch.float64), center=False,
                     onesided=False, return_complex=True)  # (B, 64, frames), bins 0..63
    ref = torch.fft.fftshift(ref, dim=1)
    ref = torch.log(ref.abs() ** 2 + 1e-6)
    assert ours.shape == ref.shape == (4, 64, L // 4), (ours.shape, ref.shape)
    check(f'ConvSTFT vs torch.stft L={L} (log power; weights stored fp32)', (ours - ref).abs().max().item(), 1e-4)
    ours32 = ConvSTFT(64, 4)(x.float()).double()
    check(f'ConvSTFT fp32 path vs torch.stft L={L}', (ours32 - ref).abs().max().item(), 1e-3)

# 2. Global DFT conv vs torch.fft.fft
g = GlobalSpectrum(384).double()
for L in (512, 1024):
    x = torch.randn(4, 2, L, dtype=torch.float64)
    sig = g._logpow(x, getattr(g, f'sig_{L}').double(), 256, L)
    X = torch.fft.fft(torch.complex(x[:, 0], x[:, 1]))
    idx = torch.arange(-128, 128) % L
    ref = torch.log(X[:, idx].abs() ** 2 / L + 1e-6)
    check(f'Global signal spectrum vs fft L={L}', (sig - ref).abs().max().item(), 1e-4)
    e = x[:, :1] ** 2 + x[:, 1:] ** 2
    e = e - e.mean(-1, keepdim=True)
    env = g._logpow(e, getattr(g, f'env_{L}').double(), 128, L)
    E = torch.fft.fft(e[:, 0])
    ref = torch.log(E[:, 1:129].abs() ** 2 / L + 1e-6)
    check(f'Envelope spectrum vs fft L={L}', (env - ref).abs().max().item(), 1e-4)

# 3. RepBlock fusion (block-level, then whole model)
blk = RepBlock(32)
blk.train()
for _ in range(20):
    blk(torch.randn(8, 32, 64) * 3 + 1)
for bn in (blk.b7, blk.b3, blk.bid):
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.uniform_(-0.5, 0.5)
blk.eval()
x = torch.randn(4, 32, 64)
y0 = blk(x)
blk.fuse()
check('RepBlock fuse', (blk(x) - y0).abs().max().item(), 1e-4)

model = TFUniEM().to(dev)
model.train()
with torch.no_grad():
    for L in (512, 1024):
        for _ in range(3):
            model(torch.randn(16, L, 2, device=dev) * 0.01, torch.randint(0, 3, (16,), device=dev))
model.eval()
fused = copy.deepcopy(model).fuse()
for L in (512, 1024):
    x = torch.randn(8, L, 2, device=dev) * 0.01
    t = torch.randint(0, 3, (8,), device=dev)
    with torch.no_grad():
        a, b = model(x, t), fused(x, t)
    shapes = {k: tuple(v.shape) for k, v in a.items()}
    print(f'forward L={L}: outputs {shapes}')
    check(f'Whole-model fuse L={L}', max((a[k] - b[k]).abs().max().item() for k in a), 1e-3)
    # RMS normalisation: output invariant to input scale (up to s feature) -> check finite
    assert all(torch.isfinite(v).all() for v in a.values())

# 4. parameter count / shared fraction
pc = count_parameters(model)
pf = count_parameters(fused)
print(f"params (train form): total {pc['total'] / 1e6:.2f}M shared {pc['shared'] / 1e6:.2f}M "
      f"shared fraction {pc['shared_fraction']:.4f}; per task used-shared fraction "
      + ', '.join(f"{k}={v['shared_fraction_of_used']:.4f}" for k, v in pc['per_task'].items()))
print(f"params (fused/inference form): total {pf['total'] / 1e6:.2f}M shared fraction {pf['shared_fraction']:.4f}")
min_shared = min(pc['shared_fraction'], pf['shared_fraction'],
                 *[v['shared_fraction_of_used'] for v in pc['per_task'].values()])
check('shared fraction > 0.80 (red line); value shown is 0.80 - min fraction', 0.80 - min_shared, 0.0)
print('ALL PASS' if ok else 'SOME TESTS FAILED')
sys.exit(0 if ok else 1)

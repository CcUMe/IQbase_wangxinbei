"""Two-GPU training, single-GPU vocabulary, two-GPU finetuning pipeline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from training.iterative_pretrain_pipeline import main


if __name__ == '__main__':
    main(default_config=ROOT / 'code/configs/pipeline/shared_base_iterative_dual.json5', required_processes=2)

"""CPU-only structural vocabulary quota and usage audit."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.vocabulary_audit import audit_vocabulary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--codebook',type=Path,required=True)
    args=parser.parse_args();report=audit_vocabulary(args.codebook,args.data)
    print('VOCABULARY AUDIT PASSED:',args.codebook,report['bpe_tokens_by_group_verified'],flush=True)

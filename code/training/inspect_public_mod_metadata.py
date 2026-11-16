"""Join public diagnostic metadata only by exact, complete IQ equality.

Public EM-Bench records are never added to training, used as model inputs, or
used to select weights. A failed match supplies no metadata for C01. Numeric
IDs or matching class names alone are not evidence of common provenance.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import urllib.request

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('artifacts/data_full'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    public = []
    for offset in [0, 100, 200]:
        path = args.output / f'rows_{offset}.json'
        if not path.exists():
            url = ('https://datasets-server.huggingface.co/rows?dataset=eye1patch%2FEM-Bench'
                   f'&config=default&split=MOD&offset={offset}&length=100')
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        payload = json.loads(path.read_text())
        assert payload['num_rows_total'] == 300
        public.extend(item['row'] for item in payload['rows'])
    assert len(public) == 300
    meta = np.load(args.data / 'metadata.npz', allow_pickle=False)
    iq = np.load(args.data / 'iq.npy', mmap_mode='r')
    # Candidate lookup accelerates equality checking; full-length equality is
    # mandatory. No rescaling, rounding, nearest neighbours, or ID inference.
    rows = np.flatnonzero((meta['modulation'] >= 0) & (meta['modulation'] < 14))
    lookup = defaultdict(list)
    for row in rows:
        lookup[iq[row, :8].tobytes()].append(int(row))
    matches = []
    for record in public:
        x = np.asarray(record['samples'], dtype=np.float32)
        if x.shape != (1, 1024, 2):
            raise ValueError(f'Unexpected public IQ shape {x.shape}')
        x = x[0]
        candidates = lookup.get(x[:8].tobytes(), [])
        for row in candidates:
            if meta['length'][row] != len(x) or not np.array_equal(iq[row, :len(x)], x):
                continue
            answer = record['answer'].strip()
            option = re.search(r'(?:^|[\s.])' + re.escape(answer) + r':\s*([^\n]+?)(?=\s+[A-E]:|\n|$)', record['human_prompt'])
            matches.append(dict(row=row, sample_id=str(meta['sample_id'][row]),
                split=int(meta['split'][row]), modulation=int(meta['modulation'][row]),
                public_id=record['id'], public_answer=option.group(1).strip() if option else None,
                reported_snr_db=float(record['snr']), reported_sample_rate=record['sample_rate'],
                equality='all 2048 float32 values, exact equality'))
    result = dict(public_source='https://huggingface.co/datasets/eye1patch/EM-Bench',
        public_records=len(public), matched_records=len({r['public_id'] for r in matches}),
        c01_matching_rows=len(matches), matches=matches,
        role='Diagnostic only; metadata never fed to models or training/sampling',
        caveats=['Reported SNR originates in public annotations, not independent measurement',
                 'Matched subset is tiny and not representative of full C01 validation',
                 'No extrapolation of reported sampling rate or SNR to unmatched records'])
    (args.output / 'exact_matches.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print({k:v for k,v in result.items() if k != 'matches'}, flush=True)


if __name__ == '__main__':
    main()

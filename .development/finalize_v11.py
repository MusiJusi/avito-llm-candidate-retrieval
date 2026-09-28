"""Freeze the locally selected v11 model and, later, its generated answer."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'artifacts/bge-m3-v11'
FINGERPRINT = '1d373ec944400cdf'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(answer=False):
    ranker = CACHE / f'final_metadata_normal_{FINGERPRINT}.joblib'
    reference = CACHE / f'benchmark_reference_{FINGERPRINT}.joblib'
    output = ROOT / 'experiments/results/v11/answer.csv'
    assert ranker.is_file() and reference.is_file()
    if answer:
        assert output.is_file()
    manifest = {
        'version': 'v11',
        'block': 'metadata',
        'recipe': 'normal',
        'weight': 0.5,
        'trees': 600,
        'ranker': ranker.relative_to(ROOT).as_posix(),
        'ranker_sha256': sha256(ranker),
        'reference': reference.relative_to(ROOT).as_posix(),
        'reference_sha256': sha256(reference),
        'answer_file': output.relative_to(ROOT).as_posix(),
        'answer_sha256': sha256(output) if answer else None,
        'selection_report': 'artifacts/bge-m3-v11/development_metadata_normal_'+FINGERPRINT+'.json',
        'platform_recall50': None,
    }
    path = CACHE / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Frozen v11 manifest', path, 'answer_hash', manifest['answer_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-answer', action='store_true')
    args = parser.parse_args()
    freeze(answer=args.freeze_answer)

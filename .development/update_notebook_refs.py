"""Update Markdown links after the one-time Avito* → solution* rename."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'Avito(?=[A-Za-z0-9_.-]*\.ipynb)')


def update():
    paths = (list(ROOT.glob('*.md')) + list((ROOT/'docs').rglob('*.md')) +
             list((ROOT/'archive').rglob('*.md')) +
             list((ROOT/'.development').glob('README.md')))
    count = 0
    touched = 0
    for path in paths:
        original = path.read_text(encoding='utf-8')
        updated, matches = PATTERN.subn('solution', original)
        if matches:
            path.write_text(updated, encoding='utf-8')
            count += matches
            touched += 1
    print('Updated', count, 'notebook references in', touched, 'Markdown files')


if __name__ == '__main__':
    update()

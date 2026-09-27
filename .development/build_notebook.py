"""Development helper: turn a commented, cell-delimited source into a notebook."""
from pathlib import Path
import hashlib
import re
import nbformat

root = Path(__file__).resolve().parents[1]
source = (root / ".development" / "notebook_source.py").read_text(encoding="utf-8")
notebook_path = root / "Avito.ipynb"
previous = nbformat.read(notebook_path, as_version=4) if notebook_path.exists() and notebook_path.stat().st_size else None
cells = []
for part in re.split(r"(?m)^# %%", source)[1:]:
    header, body = part.split("\n", 1)
    if "[markdown]" in header:
        body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
    # Stable IDs keep notebook diffs focused on actual edits.
    cells[-1].id = hashlib.sha256((cells[-1].cell_type + cells[-1].source).encode()).hexdigest()[:12]

# Markdown-only edits can retain the real execution outputs. If any executable
# source changed, clear all outputs because later cells may depend on that edit.
old_code = [c for c in previous.cells if c.cell_type == "code"] if previous else []
new_code = [c for c in cells if c.cell_type == "code"]
if [c.source for c in old_code] == [c.source for c in new_code]:
    for old, new in zip(old_code, new_code):
        new.outputs = old.outputs
        new.execution_count = old.execution_count
nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.6"},
})
nbformat.write(nb, root / "Avito.ipynb")
print(f"Created notebook: {len(cells)} cells")

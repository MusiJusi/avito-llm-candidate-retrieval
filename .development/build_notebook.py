"""Development helper: turn a commented, cell-delimited source into a notebook."""
from pathlib import Path
import re
import nbformat

root = Path(__file__).resolve().parents[1]
source = (root / ".development" / "notebook_source.py").read_text(encoding="utf-8")
cells = []
for part in re.split(r"(?m)^# %%", source)[1:]:
    header, body = part.split("\n", 1)
    if "[markdown]" in header:
        body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.6"},
})
nbformat.write(nb, root / "Avito.ipynb")
print(f"Created notebook: {len(cells)} cells")

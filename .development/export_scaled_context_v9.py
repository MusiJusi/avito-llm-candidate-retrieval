"""Refit the scaled development winner using the same context feature recipe."""
from pathlib import Path
import ast

EXPORT_SCALE_DRIVER = Path(__file__).resolve()
source = EXPORT_SCALE_DRIVER.with_name('context_scale_v9.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(EXPORT_SCALE_DRIVER)
FEATURE_SETS = ALL_FEATURE_SETS

# Load only the export function, preserving the scaled experiment's namespace.
export_source = EXPORT_SCALE_DRIVER.with_name('export_context_v9.py')
tree = ast.parse(export_source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'export']
exec(compile(tree, str(export_source), 'exec'), globals())
if __name__ == '__main__':
    export()

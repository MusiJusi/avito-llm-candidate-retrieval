"""Run the independent raw-Parquet format checks against the v9 manifest."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / '.development/validate_v8.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')]

class ManifestDirectory(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('artifacts/combined-pool-v8/', 'artifacts/context-v9/')
        return node

tree = ManifestDirectory().visit(tree)
exec(compile(tree, str(source), 'exec'), globals())
if __name__ == '__main__':
    validate()

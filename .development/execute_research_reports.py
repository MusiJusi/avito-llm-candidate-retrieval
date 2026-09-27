"""Execute report cells while retaining the main inference execution report."""
from pathlib import Path
import ast
import sys

source = Path(__file__).with_name('run_validation_notebook.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
class ReportPath(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'notebook_execution.json':
            node.value = 'research_report_execution.json'
        return node
tree = ReportPath().visit(tree)
sys.argv = [str(source), 'experiments/Avito_neural_experiments.ipynb']
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())

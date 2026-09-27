"""Increase training coverage after the 4,000-context feature pilot.

The pilot lost accuracy even without new features, suggesting that its much
smaller number of query contexts mattered. This second, bounded comparison
triples the contexts and tests only geography and its combination with semantic
neighbours. It keeps the same feature history, miner and independent encoders.
Previously viewed control remains excluded from development model selection.
"""
from pathlib import Path
import ast
import hashlib
import json
import os

SCALE_DRIVER = Path(__file__).resolve()
base_source = SCALE_DRIVER.with_name('context_rank_v9.py')
tree = ast.parse(base_source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')]
__file__ = str(base_source)
exec(compile(tree, str(base_source), 'exec'), globals())
__file__ = str(SCALE_DRIVER)
PILOT_FP = CONTEXT_FP
CONTEXT_CONFIG = dict(CONTEXT_CONFIG, evaluation_contexts=12000)
CONTEXT_FP = hashlib.sha256(json.dumps({
    'pilot': PILOT_FP, 'config': CONTEXT_CONFIG,
    'scale_driver': sha256_file(SCALE_DRIVER)}, sort_keys=True).encode()).hexdigest()[:16]

# The evaluation feature producers have not changed. Reuse their immutable
# complete pools, but never reuse the smaller training sample or fitted models.
for stage, modes in [('development', ['unseen_text', 'held_context']),
                     ('control', ['unseen_text'])]:
    for mode in modes:
        old = CONTEXT_CACHE / f'{stage}_{mode}_pools_{PILOT_FP}.joblib'
        new = CONTEXT_CACHE / f'{stage}_{mode}_pools_{CONTEXT_FP}.joblib'
        if old.exists() and not new.exists():
            os.link(old, new)

# Lower blend weights test whether a new signal helps as a complement, even
# when it cannot replace the incumbent. This grid is chosen before control.
evaluation_source = ast.parse(base_source.read_text(encoding='utf-8'))
evaluation_node = next(node for node in evaluation_source.body
                       if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
class BlendGrid(ast.NodeTransformer):
    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == 'weight':
            node.iter = ast.List(elts=[ast.Constant(v) for v in [.25, .5, .75, 1.]], ctx=ast.Load())
        return self.generic_visit(node)
evaluation_node = BlendGrid().visit(evaluation_node)
evaluation_tree = ast.fix_missing_locations(ast.Module(body=[evaluation_node], type_ignores=[]))
exec(compile(evaluation_tree, str(SCALE_DRIVER), 'exec'), globals())
ALL_FEATURE_SETS = FEATURE_SETS
FEATURE_SETS = {name: ALL_FEATURE_SETS[name] for name in ['geography', 'combined']}

if __name__ == '__main__':
    print('Scaled context experiment', CONTEXT_FP, CONTEXT_CONFIG, flush=True)
    comparison()
    path = CONTEXT_CACHE / 'selection.json'
    report = json.loads(path.read_text())
    report['limitations'] = [value.replace('4000-context', '12000-context') for value in report['limitations']]
    report['training_contexts'] = CONTEXT_CONFIG['evaluation_contexts']
    report['blend_grid'] = [.25, .5, .75, 1.]
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')

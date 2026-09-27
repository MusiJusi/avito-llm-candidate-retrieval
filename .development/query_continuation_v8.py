"""Ablate explicit hard negatives against an otherwise identical extra epoch.

Mining, sampling, seed, checkpoint and optimization remain identical. Only the
auxiliary hard-negative coefficient becomes zero. This distinguishes the effect
of extra training from the effect of the additional negative documents.
"""
from pathlib import Path
import ast

source = Path(__file__).with_name('query_hard_negative_v8.py').resolve()
tree = ast.parse(source.read_text(encoding='utf-8'))
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HN_CONFIG' for t in node.targets):
        for keyword in node.value.keywords:
            if keyword.arg=='hard_weight':keyword.value=ast.Constant(value=0.)
__file__ = str(source)
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),globals())

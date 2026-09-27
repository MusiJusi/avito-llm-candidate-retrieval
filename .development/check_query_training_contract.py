"""Check multi-positive targets without loading data, models or starting training."""
from pathlib import Path
import ast
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.inspection_deps'))
import numpy as np
source=ROOT/'.development/query_encoder_pilot.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='in_batch_targets')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),globals())
# q1 has two valid positives in the batch: neither may become a false negative.
target=in_batch_targets(['q1','q2','q3'],['a','b','c'],
    {'q1':{'a','b'},'q2':{'b'},'q3':{'c'}})
assert np.array_equal(target,np.array([[.5,.5,0],[0,1,0],[0,0,1]],np.float32))
# Duplicate documents have the same relevance; distribute weight across copies.
duplicate=in_batch_targets(['q1','q2'],['a','a'],{'q1':{'a'},'q2':{'a'}})
assert np.array_equal(duplicate,np.full((2,2),.5,np.float32))
try:
    in_batch_targets(['q1'],['negative'],{'q1':{'positive'}})
except AssertionError:pass
else:raise AssertionError('Training must reject a query with no batch positive')
print('Multi-positive masks and duplicate-document targets passed.')

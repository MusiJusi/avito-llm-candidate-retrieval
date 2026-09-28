"""Load warm-history functions into an initialized numerical namespace.

Avoid importing another complete training driver (which would duplicate indexes
and large DataFrames). Numerical recipes are read from their single source.
"""
import ast
from pathlib import Path


def activate(namespace):
    driver=Path(namespace['ROOT'])/'.development/warm_history_v10.py'
    namespace['WARM_DRIVER']=driver
    from quality_trace_v10 import QueryItemEvidence,QualityEvidenceWithTrace,TRACE_NAMES
    namespace.update(QueryItemEvidence=QueryItemEvidence,QualityEvidenceWithTrace=QualityEvidenceWithTrace,TRACE_NAMES=TRACE_NAMES)
    namespace['BASE_V10_FP']=namespace['V10_FP']
    namespace['BASE_V10_CACHE']=namespace['V10_CACHE']
    tree=ast.parse(driver.read_text(encoding='utf-8'))
    names={'WARM_CACHE','WARM_CONFIG','WARM_FP','WARM_RECIPE'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(driver),'exec'),namespace)
    namespace['WARM_CACHE'].mkdir(exist_ok=True)
    namespace['MODEL_RECIPES']['warm_mixed']=namespace['WARM_RECIPE']
    namespace['MODEL_RECIPES']['warm_deeper']={
        'columns':list(range(86)),'trees':600,'leaves':63,'min_child':150}
    namespace['MODEL_RECIPES']['quality_without_pop']={
        'columns':[c for c in range(82) if c not in {75,76,77}],
        'trees':500,'leaves':31,'min_child':80}
    import os
    namespace['os']=os
    original=namespace['fit_quality']
    def fit_with_warm(data,stage,names):
        if not any(name.startswith('warm_') for name in names):return original(data,stage,names)
        assert len(names)==1,'Fit the selected final warm model separately'
        mixed=namespace['mixed_training'](stage)
        return original(mixed,stage,names)
    namespace['fit_quality']=fit_with_warm
    return namespace

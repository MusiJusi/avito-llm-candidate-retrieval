"""Load the existing retrievers for bounded experiments without altering answer.csv.

This helper executes initialization and function definitions only. It never runs
old model selection or submission export. The final selected implementation will
be incorporated into the self-contained notebook.
"""
import ast
import json
from pathlib import Path

def definitions(source, namespace):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, '<existing notebook definitions>', 'exec'), namespace)

def bootstrap(namespace):
    snapshot = Path('archive/notebooks/Avito_v0.3.ipynb')
    if not snapshot.exists():
        snapshot = Path('Avito_v0.3.ipynb')  # Compatibility with older release archives.
    document = json.loads((snapshot if snapshot.exists() else Path('Avito.ipynb')).read_text(encoding='utf-8'))
    cells = [''.join(cell['source']) for cell in document['cells'] if cell['cell_type'] == 'code']
    for position in range(6):
        print('Initialize existing notebook cell', position + 1, flush=True)
        source = cells[position]
        if position == 4 and snapshot.exists():
            source = source.replace('ROOT / "Avito.ipynb"', 'ROOT / ' + repr(snapshot.as_posix()))
        exec(compile(source, '<notebook initialization>', 'exec'), namespace)
    definitions(cells[6], namespace)
    namespace['best_config'] = json.loads((namespace['CACHE'] / 'best_config.json').read_text())
    exec(compile(cells[8].split('training_data = build_oof_training')[0], '<existing rank features>', 'exec'), namespace)
    definitions(cells[9], namespace)
    exec(compile(cells[12], '<frozen semantic index>', 'exec'), namespace)
    namespace['legacy_retrieve_features'] = namespace['retrieve_features']
    namespace['legacy_rank_features'] = namespace['rank_features']
    namespace['legacy_score_candidates'] = namespace['score_candidates']
    namespace['legacy_negative_sample'] = namespace['hard_negative_sample']
    namespace['LEGACY_FEATURE_NAMES'] = list(namespace['RANK_FEATURE_NAMES'])
    namespace['SEMANTIC_FEATURE_NAMES'] = ['e5_cosine', 'e5_geo_affinity', 'from_lexical_pool',
        'from_semantic_pool', 'log_e5_rank', 'log_e5_geo_rank']
    namespace['RANK_FEATURE_NAMES'] += namespace['SEMANTIC_FEATURE_NAMES']
    definitions(cells[13], namespace)
    namespace['retrieve_features'] = namespace['retrieve_semantic_features']
    namespace['rank_features'] = namespace['rank_semantic_features']
    namespace['score_candidates'] = namespace['score_lexical_columns']
    namespace['hard_negative_sample'] = namespace['semantic_negative_sample']
    return namespace

if __name__ == '__main__':
    bootstrap(globals())
    eligible = ranker_history[ranker_history.item_id.isin(ITEM_TO_ROW)]
    print('Available training query texts:', eligible.query_norm.nunique(), flush=True)

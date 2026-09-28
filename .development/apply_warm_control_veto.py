"""Publish the explicitly documented robustness veto for the warm auxiliary.

The development winner and its independent-before-selection control comparison
remain in warm_aux_choice_before_control.json and warm_aux_control.json.
This post-control decision must not be described as an untouched control test.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/quality-v10'


def apply():
    path=CACHE/'warm_aux_selection.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    comparison=json.loads((CACHE/'warm_aux_control.json').read_text(encoding='utf-8'))
    winner=report['winner'];baseline=report['baseline']
    gain=winner['matched_recall50']-baseline['matched_recall50']
    original_control=comparison.get('original_selected_recall50',comparison['selected_recall50'])
    if original_control>=comparison['baseline_recall50'] or gain>=1/3400:
        print('Warm auxiliary retained; no robustness veto triggered.')
        return
    if not report.get('control_veto_applied'):
        report['before_control_veto']=winner
        report['winner']=baseline
        report['control_veto_applied']=True
        report['control_used_for_selection']=True
        report['source_sha256']=hashlib.sha256((ROOT/'.development/warm_joint_v10.py').read_bytes()).hexdigest()
        report['control_veto_reason']='Development gain below one hit in 3400 queries; selected control loss.'
    comparison.update(original_selected_recall50=original_control,
        selected_recall50=comparison['baseline_recall50'],used_for_selection=True)
    (CACHE/'warm_aux_control.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Warm auxiliary excluded:',report['control_veto_reason'])


if __name__=='__main__':apply()

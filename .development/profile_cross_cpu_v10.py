"""Bounded CPU timing of float32 and BF16 on the local trained pilot."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.semantic_deps'),str(ROOT/'.inspection_deps')]
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer


def profile():
    torch.set_num_threads(8)
    selection=json.loads((ROOT/'artifacts/cross-finetune-v10/selection.json').read_text())
    model=AutoModelForSequenceClassification.from_pretrained(ROOT/selection['model'],
        local_files_only=True,attn_implementation='eager').float().eval()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/mmarco-miniLM-cross-encoder',local_files_only=True)
    batch=tokenizer(['монтаж видеодомофонов']*32,['Установка видеодомофонов. '+('Монтаж и ремонт оборудования. '*60)]*32,
        padding=True,truncation=True,max_length=192,return_tensors='pt')
    report={}
    with torch.inference_mode():
        for name,enabled in [('float32',False),('bfloat16',True)]:
            start=time.perf_counter()
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled):
                for _ in range(5):result=model(**batch).logits
            seconds=(time.perf_counter()-start)/5
            report[name]={'seconds_per_32_pairs':seconds,'pairs_per_second':32/seconds,
                'estimated_benchmark_seconds':2452*200/32*seconds}
            print(name,json.dumps(report[name]),flush=True)
    (ROOT/'artifacts/cross-finetune-v10/cpu_profile.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':profile()

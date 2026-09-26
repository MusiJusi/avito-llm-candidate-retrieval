"""Read-only dataset size and memory audit; does not modify the input files."""

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
# Keep inspection dependencies separate from the future notebook environment.
sys.path.insert(0, str(ROOT / ".inspection_deps"))

import pandas as pd
import psutil
import pyarrow.parquet as pq

process = psutil.Process()
report = {
    "versions": {"python": sys.version, "pandas": pd.__version__},
    "ram_total_bytes": psutil.virtual_memory().total,
    "ram_available_bytes": psutil.virtual_memory().available,
    "disk_free_bytes": psutil.disk_usage(str(ROOT)).free,
    "cpu_physical_cores": psutil.cpu_count(logical=False),
    "cpu_logical_cores": psutil.cpu_count(),
    "baseline_rss_bytes": process.memory_info().rss,
    "files": [],
}
frames = []
for name in ["benchmark_queries.parquet", "benchmark_items.parquet", "train.parquet"]:
    path = ROOT / name
    metadata = pq.ParquetFile(path).metadata
    started = time.perf_counter()
    frame = pd.read_parquet(path)
    elapsed = time.perf_counter() - started
    frames.append(frame)
    memory = frame.memory_usage(deep=True)
    info = {
        "name": name,
        "disk_bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": len(frame.columns),
        "row_groups": metadata.num_row_groups,
        "parquet_uncompressed_bytes": sum(
            metadata.row_group(i).total_byte_size
            for i in range(metadata.num_row_groups)
        ),
        "read_seconds": round(elapsed, 3),
        "dataframe_bytes": int(memory.sum()),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "largest_columns_bytes": {
            column: int(size) for column, size in memory.nlargest(6).items()
        },
        "text_lengths_chars": {},
    }
    for column in ["search_query", "item_title_raw", "item_description_raw", "item_infm_params_text"]:
        if column in frame:
            lengths = frame[column].str.len()
            info["text_lengths_chars"][column] = {
                "nulls": int(frame[column].isna().sum()),
                "mean": round(float(lengths.mean()), 1),
                "p50": float(lengths.quantile(0.5)),
                "p95": float(lengths.quantile(0.95)),
                "p99": float(lengths.quantile(0.99)),
                "max": float(lengths.max()),
            }
    for column in ["item_id", "query_id"]:
        if column in frame:
            info[column + "_unique"] = int(frame[column].nunique())
    # Older pandas environments often store text as Python objects. Measure
    # that alternative one column at a time without retaining another dataset.
    object_bytes = 0
    for column in frame:
        series = frame[column]
        if pd.api.types.is_string_dtype(series.dtype):
            object_bytes += int(series.astype(object).memory_usage(index=False, deep=True))
        else:
            object_bytes += int(series.memory_usage(index=False, deep=True))
    info["dataframe_with_object_strings_bytes"] = object_bytes + int(frame.index.memory_usage(deep=True))
    gc.collect()
    report["files"].append(info)
    print(json.dumps(info, ensure_ascii=True), flush=True)

report["combined_dataframe_bytes"] = sum(f["dataframe_bytes"] for f in report["files"])
report["combined_object_dataframe_bytes"] = sum(f["dataframe_with_object_strings_bytes"] for f in report["files"])
report["final_rss_bytes"] = process.memory_info().rss
report["process_memory"] = process.memory_info()._asdict()
(ROOT / "data_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=True), flush=True)

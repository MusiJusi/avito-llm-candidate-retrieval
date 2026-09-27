"""Execute ordinary Python notebook cells in one namespace, preserving outputs.

No notebook-server installation or open network port is needed. The submitted
notebook contains the complete solution and does not depend on this helper.
"""
import io
import os
from pathlib import Path
import sys
import time
import traceback
import types
import nbformat

# Reproduction can explicitly reject network connections while executing cells.
# Model loading must work from local files, not from an existing HF online cache.
if os.environ.get("AVITO_DISABLE_NETWORK") == "1":
    import socket
    def reject_network(*args, **kwargs):
        raise RuntimeError("Network access is disabled for offline reproduction")
    socket.socket.connect = reject_network
    socket.socket.connect_ex = reject_network
    socket.create_connection = reject_network

root = Path(__file__).resolve().parents[1]
os.chdir(root)
nb = nbformat.read(root / "Avito.ipynb", as_version=4)
module = types.ModuleType("__main__")
sys.modules["__main__"] = module
namespace = module.__dict__

class Tee(io.StringIO):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def write(self, value):
        self.original.write(value)
        self.original.flush()
        return super().write(value)
    def flush(self):
        self.original.flush()

count = 0
for position, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    count += 1
    print(f"\nCELL {position + 1}/{len(nb.cells)} (execution {count})", flush=True)
    started = time.perf_counter()
    stdout, stderr = sys.stdout, sys.stderr
    captured_out, captured_err = Tee(stdout), Tee(stderr)
    cell.outputs = []
    cell.execution_count = count
    error = None
    try:
        sys.stdout, sys.stderr = captured_out, captured_err
        exec(compile(cell.source, f"Avito.ipynb:cell-{position + 1}", "exec"), namespace)
    except BaseException as exc:
        error = exc
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        for name, captured in [("stdout", captured_out), ("stderr", captured_err)]:
            if captured.getvalue():
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=captured.getvalue()))
        nbformat.write(nb, root / "Avito.ipynb")
    print(f"Cell elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    if error is not None:
        raise error

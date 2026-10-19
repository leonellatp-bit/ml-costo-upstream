"""Convierte scripts en formato 'percent' (# %% / # %% [markdown]) a .ipynb y los ejecuta.

Uso:
    python tools/nbtools.py build book/src/01_eda.py book/01_eda.ipynb   # convierte
    python tools/nbtools.py run book/01_eda.ipynb                         # ejecuta in situ

``run`` es un ejecutor mínimo (sin servidor Jupyter): ejecuta las celdas en un
espacio de nombres compartido y guarda stdout, el valor de la última expresión
(tablas pandas en HTML) y las figuras de matplotlib como PNG. En una máquina con
Jupyter se puede usar en su lugar ``jupyter nbconvert --execute`` o
``jupyter-book build book/``.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _cell(kind, src):
    src = src.strip("\n")
    lines = [l + "\n" for l in src.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build(py_path, ipynb_path):
    text = Path(py_path).read_text(encoding="utf-8")
    cells = []
    kind, buf = None, []
    for line in text.split("\n"):
        if line.startswith("# %%"):
            if kind is not None and "".join(buf).strip():
                cells.append((kind, "\n".join(buf)))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            if kind == "markdown":
                line = line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
            buf.append(line)
    if kind is not None and "".join(buf).strip():
        cells.append((kind, "\n".join(buf)))
    nb = {"cells": [_cell(k, s) for k, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(ipynb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def _display_data(obj):
    import pandas as pd
    if obj is None:
        return None
    if hasattr(obj, "_repr_html_") and isinstance(obj, (pd.DataFrame,)) or hasattr(obj, "to_html") and hasattr(obj, "data") and type(obj).__name__ == "Styler":
        html = obj.to_html() if type(obj).__name__ == "Styler" else obj._repr_html_()
        txt = repr(obj) if not type(obj).__name__ == "Styler" else ""
        return {"text/html": html.split("\n"), "text/plain": txt.split("\n")}
    if type(obj).__module__.startswith("matplotlib"):
        return None
    if type(obj).__name__ == "Markdown":
        return {"text/markdown": obj.data.split("\n"), "text/plain": ["<Markdown>"]}
    return {"text/plain": repr(obj).split("\n")}


def _lines(s):
    out = s.split("\n")
    return [l + "\n" for l in out[:-1]] + ([out[-1]] if out[-1] else [])


def run(ipynb_path, cwd=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(ipynb_path).resolve()
    nb = json.loads(path.read_text(encoding="utf-8"))
    os.chdir(cwd or path.parent)
    sys.path.insert(0, str(path.parent.parent))
    ns = {"__name__": "__main__"}
    outputs_holder = []

    class Markdown:
        def __init__(self, data):
            self.data = data

    def display(*objs):
        for o in objs:
            d = _display_data(o)
            if d:
                outputs_holder.append({"output_type": "display_data", "data": d, "metadata": {}})

    ns["display"] = display
    ns["Markdown"] = Markdown
    count = 0
    t_all = time.perf_counter()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        src = "".join(cell["source"])
        outputs_holder.clear()
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            tree = ast.parse(src)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                val = eval(compile(last, f"<cell {count}>", "eval"), ns) if last else None
        except Exception:
            print(f"ERROR en celda {count}:\n{src[:400]}", file=sys.stderr)
            traceback.print_exc()
            raise
        outs = []
        if buf.getvalue():
            outs.append({"output_type": "stream", "name": "stdout", "text": _lines(buf.getvalue())})
        outs.extend(list(outputs_holder))
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                  "text/plain": [f"<Figure {num}>"]}})
        plt.close("all")
        d = _display_data(val) if last else None
        if d:
            outs.append({"output_type": "execute_result", "execution_count": count, "data": d, "metadata": {}})
        cell["outputs"] = outs
        cell["execution_count"] = count
        print(f"  celda {count:>3} ok ({time.perf_counter() - t0:5.1f}s)", flush=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path.name}: {count} celdas ejecutadas en {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        run(sys.argv[2])

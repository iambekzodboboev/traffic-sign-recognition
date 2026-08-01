"""Stage 8.1 -- attach the outputs captured by capture_local_notebook_outputs.py
into the actual committed notebooks (02_data_audit_eda.ipynb,
03_preprocessing.ipynb), so they show real results on GitHub instead of
code only. Cell *source* is left untouched; only `outputs` and
`execution_count` are written, matching what that cell's Colab-run output
would look like for the same underlying dataset (see
capture_local_notebook_outputs.py's docstring for why local, not Colab).

One-time local use, after capture_local_notebook_outputs.py has produced
data/local_notebook_outputs.pkl:
    python scripts/attach_local_notebook_outputs.py
"""
import base64
import json
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKLE_PATH = PROJECT_ROOT / "data" / "local_notebook_outputs.pkl"

NOTEBOOK_MAP = {
    "nb02": PROJECT_ROOT / "notebooks" / "02_data_audit_eda.ipynb",
    "nb03": PROJECT_ROOT / "notebooks" / "03_preprocessing.ipynb",
}


def make_outputs(entry):
    outputs = []
    if entry["stdout"]:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": entry["stdout"].splitlines(keepends=True),
        })
    if entry["repr"] is not None:
        outputs.append({
            "output_type": "execute_result",
            "execution_count": 1,
            "data": {"text/plain": entry["repr"].splitlines(keepends=True)},
            "metadata": {},
        })
    for png_bytes in entry["images"]:
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(png_bytes).decode("ascii")},
            "metadata": {},
        })
    return outputs


def main():
    with open(PICKLE_PATH, "rb") as f:
        captured = pickle.load(f)

    for nb_key, nb_path in NOTEBOOK_MAP.items():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        exec_count = 1
        attached = 0
        # Match by position: code cells in order correspond to the indices
        # used when capturing (cell_index in the real notebook's cell list).
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            if i not in captured[nb_key]:
                continue
            entry = captured[nb_key][i]
            cell["outputs"] = make_outputs(entry)
            cell["execution_count"] = exec_count
            exec_count += 1
            attached += 1

        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{nb_path.name}: attached outputs to {attached} cell(s)")


if __name__ == "__main__":
    main()

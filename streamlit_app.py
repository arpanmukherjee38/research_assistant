import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

for _p in [str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

exec(open(ROOT / "ui" / "app.py", encoding="utf-8").read())

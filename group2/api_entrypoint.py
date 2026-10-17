"""Only Python file: Uvicorn requires an importable ASGI module. Trusted local notebooks only."""
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
ROOT = Path(__file__).resolve().parent
import os
if os.environ.get("GROUP2_DEV_MODE", "false").lower() != "true":
    raise RuntimeError("Notebook dev API refuses production mode without shared Group 3 auth")
os.chdir(ROOT / "notebooks")
shell = InteractiveShell.instance()
shell.run_line_magic("run", "07_fastapi_integration.ipynb")
app = shell.user_ns["app"]

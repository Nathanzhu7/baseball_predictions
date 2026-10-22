from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

collect_ignore = ["manual"]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    return project_root / "data"


@pytest.fixture(scope="session")
def sample_screenshot_path(data_dir: Path) -> Path:
    path = data_dir / "screenshots" / "1024x768.png"
    if not path.exists():
        pytest.skip("sample screenshot missing")
    return path


@pytest.fixture()
def sample_export_csv(tmp_path: Path, project_root: Path) -> Path:
    source = project_root / "skills2.csv"
    if not source.exists():
        pytest.skip("sample export CSV missing")

    dest = tmp_path / source.name
    shutil.copy(source, dest)
    return dest

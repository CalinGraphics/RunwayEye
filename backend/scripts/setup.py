"""One-time setup on a fresh clone, on Windows, macOS or Linux. Safe to run again.

    py -3 backend/scripts/setup.py          # Windows
    python3 backend/scripts/setup.py        # macOS / Linux

1. creates venv/ (recreated if it was copied from another computer and no longer runs),
2. installs requirements.txt (on Linux, the CPU build of torch: ~200 MB instead of ~2.5 GB of CUDA),
3. copies .env.example to .env,
4. draws the demo clip used when there is no camera (data/videos/synthetic_demo.mp4).

Then: start_demo.bat (Windows) or ./start_demo.sh (macOS / Linux).
Options: --gpu keeps the default torch build on Linux (NVIDIA), --skip-install only does 3-4.
"""

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root: venv/, .env, data/
BACKEND = ROOT / "backend"
VENV = ROOT / "venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_CPU = "https://download.pytorch.org/whl/cpu"


def step(msg: str) -> None:
    print(f"\n[RunwayEye] {msg}", flush=True)


def run(*cmd: str | Path) -> None:
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def venv_works() -> bool:
    if not PY.exists():
        return False
    try:
        return subprocess.run([str(PY), "-c", "import sys"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="Linux: păstrează torch cu CUDA")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    if sys.version_info < (3, 11):
        sys.exit(f"Python {sys.version.split()[0]} e prea vechi: e nevoie de 3.11 sau mai nou (testat pe 3.14).")

    if not args.skip_install:
        if not venv_works():
            if VENV.exists():
                step("venv/ nu rulează pe acest calculator (copiat de pe altul?): îl refac.")
                shutil.rmtree(VENV)
            step(f"Creez venv/ cu Python {sys.version.split()[0]} …")
            venv.create(VENV, with_pip=True)
        run(PY, "-m", "pip", "install", "--upgrade", "pip", "-q")
        if sys.platform.startswith("linux") and not args.gpu:
            step("Instalez torch (varianta CPU) …")
            run(PY, "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_CPU)
        step("Instalez dependențele (câteva minute) …")
        run(PY, "-m", "pip", "install", "-r", BACKEND / "requirements.txt")

    env = ROOT / ".env"
    if not env.exists():
        shutil.copy(ROOT / ".env.example", env)
        step("Am creat .env din .env.example (VIDEO_SOURCE=0: prima cameră web).")

    clip = ROOT / "data/videos/synthetic_demo.mp4"
    if not clip.exists():
        step("Desenez clipul demo (folosit când nu există cameră) …")
        run(PY if PY.exists() else sys.executable, BACKEND / "scripts/make_demo_clip.py")

    model = BACKEND / "models/best.pt"
    step("Gata." + ("" if model.exists() else " ATENȚIE: models/best.pt lipsește, se folosește modelul provizoriu."))
    print("  Pornire:   " + ("start_demo.bat" if os.name == "nt" else "./start_demo.sh"))
    print(f"  Verificare: {PY.relative_to(ROOT)} backend/scripts/demo_check.py")
    print(f"  Test model: {PY.relative_to(ROOT)} backend/scripts/test_samples.py")


if __name__ == "__main__":
    main()

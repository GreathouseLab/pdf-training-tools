
#!/usr/bin/env python3
"""
setup_env.py — Reproducible environment bootstrapper

What this script does:
1) Creates two virtual environments:
   - .venv (general env for your pipeline)
   - pymupdf-venv (optional dedicated env for PyMuPDF, as in your original commands)
2) Installs packages into each env (mirrors your exact shell sequence).
3) Writes a .env file with your API keys / config (if provided).
4) Writes helper shell scripts you can `source` to activate and export vars.

IMPORTANT:
- A Python process cannot "activate" a venv for your current shell session.
  Instead, this script installs everything and writes shell scripts you can `source`
  to activate in your terminal:
    source activate_env.sh
    source activate_pymupdf.sh
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_MAIN = ROOT / ".venv"
VENV_PYMUPDF = ROOT / "pymupdf-venv"

DEFAULT_PACKAGES = [
    "pymupdf", "openai", "python-dotenv", "tqdm",
    "pinecone", "matplotlib", "pydantic", "deepseek", "together"
]

def run(cmd, env=None, cwd=None):
    print(f"[run] {cmd}")
    res = subprocess.run(cmd, shell=isinstance(cmd, str), check=True, env=env, cwd=cwd)
    return res.returncode

def pybin(venv: Path) -> Path:
    """Return the python executable path inside a venv (POSIX)."""
    return venv / "bin" / "python"

def pip_install(venv: Path, packages):
    if isinstance(packages, str):
        packages = [packages]
    cmd = [str(pybin(venv)), "-m", "pip", "install", *packages]
    run(cmd)

def pip_upgrade(venv: Path):
    run([str(pybin(venv)), "-m", "pip", "install", "--upgrade", "pip"])

def create_venv(venv: Path):
    if not venv.exists():
        print(f"[venv] Creating {venv}")
        run([sys.executable, "-m", "venv", str(venv)])
    else:
        print(f"[venv] Exists: {venv}")

def main():
    ap = argparse.ArgumentParser(description="Bootstrap virtualenvs and install packages.")
    ap.add_argument("--requirements", default="requirements.txt",
                    help="Optional requirements file (installed into pymupdf-venv to mirror your sequence).")
    ap.add_argument("--openai-key", default=None)
    ap.add_argument("--deepseek-key", default=None)
    ap.add_argument("--pinecone-key", default=None)
    ap.add_argument("--pinecone-env", default="us-west1-gcp")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--pinecone-index", default="paper-qa")
    ap.add_argument("--skip-second-venv", action="store_true",
                    help="Skip creating and installing into pymupdf-venv.")
    args = ap.parse_args()

    # 1) .venv
    create_venv(VENV_MAIN)
    pip_upgrade(VENV_MAIN)
    pip_install(VENV_MAIN, DEFAULT_PACKAGES)

    # 2) pymupdf-venv (optional)
    if not args.skip_second_venv:
        create_venv(VENV_PYMUPDF)
        pip_upgrade(VENV_PYMUPDF)
        pip_install(VENV_PYMUPDF, "--upgrade pymupdf")

        req = ROOT / args.requirements
        if req.exists():
            print(f"[req] Installing -r {req} into pymupdf-venv")
            run([str(pybin(VENV_PYMUPDF)), "-m", "pip", "install", "-r", str(req)])
        else:
            print(f"[req] Skipped: {req} not found")

    # 3) .env file + activation helpers
    lines = []
    def add_line(key, val):
        if val is None:
            lines.append(f"# {key}=")
        else:
            lines.append(f"{key}={val}")

    add_line("OPENAI_API_KEY", args.openai_key)
    add_line("DEEPSEEK_API_KEY", args.deepseek_key)
    add_line("PINECONE_API_KEY", args.pinecone_key)
    add_line("PINECONE_ENV", args.pinecone_env)
    add_line("MODEL", args.model)
    add_line("PINECONE_INDEX", args.pinecone_index)

    (ROOT / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {(ROOT / '.env')}")

    activate_env_sh = f"""# Usage: source activate_env.sh
source "{VENV_MAIN}/bin/activate"
export $(grep -v '^#' .env | xargs -I{{}} echo {{}})
echo "[activated] .venv with env vars from .env"
"""
    (ROOT / "activate_env.sh").write_text(activate_env_sh, encoding="utf-8")
    print(f"[write] {(ROOT / 'activate_env.sh')}")

    activate_pymupdf_sh = f"""# Usage: source activate_pymupdf.sh
source "{VENV_PYMUPDF}/bin/activate"
export $(grep -v '^#' .env | xargs -I{{}} echo {{}})
echo "[activated] pymupdf-venv with env vars from .env"
"""
    (ROOT / "activate_pymupdf.sh").write_text(activate_pymupdf_sh, encoding="utf-8")
    print(f"[write] {(ROOT / 'activate_pymupdf.sh')}")

    print("\\nDone.")
    print("Next steps:")
    print("  1) To use your main env:   source activate_env.sh")
    if not args.skip_second_venv:
        print("  2) Or PyMuPDF-only env:    source activate_pymupdf.sh")
    print("  3) .env contains your keys; python-dotenv will auto-load if your code calls load_dotenv().")

if __name__ == "__main__":
    main()

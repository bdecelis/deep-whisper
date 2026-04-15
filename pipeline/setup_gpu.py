"""
deep-whisper · pipeline/setup_gpu.py
=====================================
GPU stack installer. Registered as the `deep-whisper-setup` console script
so it is available immediately after `pip install deep-whisper`.

Handles everything that pip install cannot safely do on its own:
  - Detects the correct CUDA version for this Python environment
  - Installs PyTorch with the matching CUDA build if needed
  - Fixes pytorch-lightning < 2.0.0 (invalid pip 24.1+ metadata)
  - Installs faster-whisper
  - Installs whisperx, then detects and repairs any torch CUDA breakage
  - Copies cuDNN 8 DLLs into the CTranslate2 package dir (Windows only)
  - Runs a final verification and reports clearly

Usage:
    deep-whisper-setup              # after pip install deep-whisper
    python -m pipeline.setup_gpu   # alternative invocation
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from typing import Optional


TAG = "[deep-whisper-setup]"
IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def clean_env() -> dict:
    """os.environ with PYTHONNOUSERSITE=1 to isolate pip from user site-packages."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_pip(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "pip", *args, "--no-user"]
    return subprocess.run(
        cmd,
        env=clean_env(),
        capture_output=capture,
        text=capture,
    )


def run_py(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=clean_env(),
    )


def is_installed(dist_name: str, min_version: str = "0") -> bool:
    try:
        from packaging.version import Version
        v = importlib.metadata.version(dist_name)
        return Version(v) >= Version(min_version)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 0 — pytorch-lightning pre-flight
# ---------------------------------------------------------------------------

def fix_pytorch_lightning() -> None:
    """Upgrade/remove pytorch-lightning < 2.0.0 before any pip resolution."""
    result = run_pip("show", "pytorch-lightning", capture=True)
    if result.returncode != 0:
        print(f"{TAG}   pytorch-lightning not present — OK")
        return

    version = ""
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
    if not version:
        return

    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        major = 0

    if major >= 2:
        print(f"{TAG}   pytorch-lightning {version} — OK")
        return

    print(f"{TAG}   pytorch-lightning {version} has invalid metadata — upgrading ...")
    upgrade = run_pip("install", "pytorch-lightning>=2.0.0", capture=True)
    if upgrade.returncode == 0:
        print(f"{TAG}   pytorch-lightning upgraded.")
    else:
        print(f"{TAG}   Upgrade failed — removing instead ...")
        run_pip("uninstall", "pytorch-lightning", "-y")
        print(f"{TAG}   pytorch-lightning removed.")


# ---------------------------------------------------------------------------
# Step 1 — detect CUDA version for this environment
# ---------------------------------------------------------------------------

def detect_cuda() -> Optional[str]:
    """
    Return the CUDA version string for this Python environment.

    Priority:
      1. torch.version.cuda   — correct for existing envs (ComfyUI, venvs)
      2. nvidia-smi           — driver ceiling for fresh environments
      3. NVML via ctypes      — nvidia-smi fallback, no PATH required
    Returns None if CUDA is not detectable.
    """
    # 1. torch.version.cuda
    r = run_py("import torch; print(torch.version.cuda or 'cpu')")
    if r.returncode == 0:
        v = r.stdout.strip()
        if v not in ("cpu", "None", ""):
            return v

    # 2. nvidia-smi
    for smi in (["nvidia-smi"],
                [r"C:\Windows\System32\nvidia-smi.exe"] if IS_WINDOWS else []):
        try:
            r2 = subprocess.run(smi, capture_output=True, text=True, timeout=10)
            if r2.returncode == 0:
                import re
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r2.stdout)
                if m:
                    return m.group(1)
        except Exception:
            pass

    # 3. NVML via ctypes (Windows)
    if IS_WINDOWS:
        r3 = run_py(
            "import ctypes, sys\n"
            "nvml=ctypes.WinDLL('nvml.dll')\n"
            "nvml.nvmlInit_v2()\n"
            "v=ctypes.c_int()\n"
            "nvml.nvmlSystemGetCudaDriverVersion(ctypes.byref(v))\n"
            "print(str(v.value//1000)+'.'+str((v.value%1000)//10))\n"
            "nvml.nvmlShutdown()"
        )
        if r3.returncode == 0 and r3.stdout.strip():
            return r3.stdout.strip()

    return None


def cuda_to_tag(cuda_version: str) -> Optional[str]:
    """Map a CUDA version string (e.g. '12.8') to a PyTorch wheel tag."""
    try:
        major, minor = int(cuda_version.split(".")[0]), int(cuda_version.split(".")[1])
    except (ValueError, IndexError):
        return None
    if   major >= 12 and minor >= 8: return "cu128"
    elif major >= 12 and minor >= 6: return "cu126"
    elif major >= 12 and minor >= 4: return "cu124"
    elif major >= 12 and minor >= 1: return "cu121"
    elif major >= 11 and minor >= 8: return "cu118"
    return None


# ---------------------------------------------------------------------------
# Step 2 — ensure PyTorch CUDA is installed
# ---------------------------------------------------------------------------

def ensure_torch(cuda_tag: str) -> bool:
    """Install or verify torch with the given CUDA tag. Returns True on success."""
    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"

    r = run_py("import torch; print(torch.version.cuda or 'cpu')")
    if r.returncode == 0 and r.stdout.strip() not in ("cpu", "None", ""):
        print(f"{TAG}   torch already installed with CUDA {r.stdout.strip()} — OK")
        return True

    print(f"{TAG}   Installing torch + torchaudio ({cuda_tag}) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-user",
         "torch", "torchaudio", "--index-url", index_url],
        env=clean_env(),
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Step 3 — torch state capture / restore
# ---------------------------------------------------------------------------

def get_torch_state() -> Optional[dict]:
    r = run_py(
        "import torch; "
        "print(torch.__version__); "
        "print(torch.version.cuda or 'None'); "
        "print(torch.cuda.is_available())"
    )
    if r.returncode != 0:
        return None
    lines = r.stdout.strip().splitlines()
    if len(lines) < 3:
        return None
    return {
        "version":        lines[0].strip(),
        "cuda_version":   lines[1].strip(),
        "cuda_available": lines[2].strip() == "True",
    }


def restore_torch(state: dict) -> None:
    tag = cuda_to_tag(state["cuda_version"])
    if not tag:
        print(f"{TAG}   Cannot determine index URL — manual torch reinstall required.")
        return
    base    = state["version"].split("+")[0]
    index   = f"https://download.pytorch.org/whl/{tag}"
    print(f"{TAG}   Restoring torch=={base} from {index} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-user",
         "--force-reinstall", "--index-url", index,
         f"torch=={base}", f"torchaudio=={base}"],
        env=clean_env(),
    )


# ---------------------------------------------------------------------------
# Step 4 — cuDNN 8 DLL placement (Windows only)
# ---------------------------------------------------------------------------

def setup_cudnn(cuda_tag: str) -> None:
    if not IS_WINDOWS:
        return

    pkg      = "nvidia-cudnn-cu11==8.9.7.29" if cuda_tag == "cu118" else "nvidia-cudnn-cu12==8.9.7.29"
    required = ["cudnn64_8.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll"]

    # Find ctranslate2 dir
    r = run_py("import ctranslate2, os; print(os.path.dirname(ctranslate2.__file__))")
    if r.returncode != 0:
        print(f"{TAG}   ctranslate2 not installed yet — cuDNN setup deferred.")
        return
    ct2_dir = r.stdout.strip()

    # Check if DLLs already present
    if all((os.path.exists(os.path.join(ct2_dir, d))) for d in required):
        print(f"{TAG}   cuDNN DLLs already present — OK")
        return

    print(f"{TAG}   Installing {pkg} ...")
    run_pip("install", pkg)

    # Find installed DLLs
    r2 = run_py("import site; print(site.getsitepackages()[0])")
    if r2.returncode != 0:
        print(f"{TAG}   Could not locate site-packages — skipping cuDNN copy.")
        return
    cudnn_bin = os.path.join(r2.stdout.strip(), "nvidia", "cudnn", "bin")

    copied  = 0
    missing = []
    for dll in required:
        src = os.path.join(cudnn_bin, dll)
        dst = os.path.join(ct2_dir, dll)
        if os.path.exists(dst):
            copied += 1
            continue
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"{TAG}   Copied {dll} -> ctranslate2/")
            copied += 1
        else:
            missing.append(dll)

    if missing:
        print(f"{TAG}   WARNING: could not find: {', '.join(missing)}")
        print(f"{TAG}   Manual fix: copy cuDNN v8 bin/*.dll to {ct2_dir}")
    else:
        print(f"{TAG}   cuDNN DLLs in place ({copied}/{len(required)})")


# ---------------------------------------------------------------------------
# Step 5 — install faster-whisper and whisperx (with torch protection)
# ---------------------------------------------------------------------------

def install_gpu_deps(torch_before: Optional[dict]) -> None:
    if not is_installed("faster_whisper", "1.0.0"):
        print(f"{TAG}   Installing faster-whisper ...")
        run_pip("install", "faster-whisper>=1.0.0")
    else:
        print(f"{TAG}   faster-whisper already installed — OK")

    print(f"{TAG}   Installing whisperx ...")
    run_pip("install", "whisperx>=3.0.0")

    # Repair torch if whisperx broke it
    if torch_before and torch_before["cuda_available"]:
        after = get_torch_state()
        broken = (
            after is None
            or not after["cuda_available"]
            or after["cuda_version"] != torch_before["cuda_version"]
        )
        if broken:
            print(f"{TAG}   torch CUDA overwritten by whisperx — restoring ...")
            restore_torch(torch_before)
            final = get_torch_state()
            if final and final["cuda_available"]:
                print(f"{TAG}   torch CUDA restored ({final['version']} / CUDA {final['cuda_version']})")
            else:
                print(f"{TAG}   ERROR: could not restore torch automatically.")
                print(f"{TAG}   Run: pip install --force-reinstall --no-user "
                      f"--index-url https://download.pytorch.org/whl/"
                      f"{cuda_to_tag(torch_before['cuda_version'])} "
                      f"torch torchaudio")
        else:
            print(f"{TAG}   torch CUDA intact after whisperx — OK")


# ---------------------------------------------------------------------------
# Step 6 — verification
# ---------------------------------------------------------------------------

def verify() -> bool:
    """Import the key modules and verify CUDA works. Returns True if all pass."""
    checks = [
        ("torch + CUDA",     "import torch; assert torch.cuda.is_available(), 'CUDA not available'"),
        ("faster-whisper",   "import faster_whisper"),
        ("whisperx",         "import whisperx"),
        ("silero-vad",       "from silero_vad import load_silero_vad"),
        ("librosa",          "import librosa"),
    ]
    all_ok = True
    for label, code in checks:
        r = run_py(code)
        if r.returncode == 0:
            print(f"  OK    {label}")
        else:
            err = (r.stderr + r.stdout).strip().splitlines()
            print(f"  FAIL  {label}")
            if err:
                print(f"        {err[-1]}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 56)
    print("  deep-whisper GPU setup")
    print("=" * 56)
    print()

    # 0. Pre-flight
    print(f"[0/5] Checking for known dependency conflicts ...")
    fix_pytorch_lightning()

    # 1. CUDA detection
    print(f"\n[1/5] Detecting CUDA version ...")
    cuda_version = detect_cuda()
    if cuda_version is None:
        print(f"{TAG} Could not detect CUDA version automatically.")
        print(f"{TAG} To find it: python -c \"import torch; print(torch.version.cuda)\"")
        print(f"{TAG} Or run: nvidia-smi")
        print()
        tag_input = input("  Enter your CUDA tag (e.g. cu128): ").strip()
        if not tag_input.startswith("cu"):
            print("Invalid tag.")
            return 1
        cuda_tag = tag_input
    else:
        cuda_tag = cuda_to_tag(cuda_version)
        if cuda_tag is None:
            print(f"{TAG} CUDA {cuda_version} is below the minimum supported version (11.8).")
            return 1
        print(f"{TAG}   CUDA {cuda_version} → PyTorch tag: {cuda_tag}")

    # 2. PyTorch
    print(f"\n[2/5] Ensuring PyTorch CUDA ({cuda_tag}) ...")
    torch_ok = ensure_torch(cuda_tag)
    if not torch_ok:
        print(f"{TAG} PyTorch installation failed.")
        return 1

    # 3. Capture torch state, install GPU deps
    print(f"\n[3/5] Installing GPU pipeline dependencies ...")
    torch_before = get_torch_state()
    install_gpu_deps(torch_before)

    # 4. cuDNN (Windows only)
    print(f"\n[4/5] Setting up cuDNN (Windows) ...")
    setup_cudnn(cuda_tag)

    # 5. Verify
    print(f"\n[5/5] Verifying installation ...")
    ok = verify()

    print()
    print("=" * 56)
    if ok:
        print("  Setup complete. Run test_env.py to confirm end-to-end.")
    else:
        print("  Setup completed with errors — see above.")
    print("=" * 56)
    print()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

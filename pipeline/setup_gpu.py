"""
deep-whisper · pipeline/setup_gpu.py
=====================================
GPU stack installer. Registered as the `deep-whisper-setup` console script
so it is available immediately after `pip install deep-whisper`.

Can always be invoked directly without PATH:
    python -m deep_whisper.pipeline.setup_gpu

Usage:
    deep-whisper-setup                   # if Scripts/ is on PATH
    python -m deep_whisper.pipeline.setup_gpu         # always works
    python_embeded\\Scripts\\deep-whisper-setup.exe  # ComfyUI portable full path
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Optional


TAG = "[deep-whisper-setup]"
IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def clean_env() -> dict:
    """os.environ with PYTHONNOUSERSITE=1."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_pip(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", *args, "--no-user"],
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
        return Version(importlib.metadata.version(dist_name)) >= Version(min_version)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 0 — pytorch-lightning pre-flight
# ---------------------------------------------------------------------------

def fix_pytorch_lightning() -> None:
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
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-user", "pytorch-lightning>=2.0.0"],
        env=clean_env(), capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"{TAG}   pytorch-lightning upgraded.")
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "pytorch-lightning", "-y"],
            env=clean_env(), check=False,
        )
        print(f"{TAG}   pytorch-lightning removed.")


# ---------------------------------------------------------------------------
# Step 1 — detect CUDA version for this environment
# ---------------------------------------------------------------------------

def detect_cuda() -> Optional[str]:
    """
    Return the CUDA version string for this Python environment.

    Priority:
      1. torch.version.cuda AND torch.cuda.is_available() — only trust this
         if CUDA actually works, not just if the version string is present.
      2. torch.version.cuda alone — torch is a CUDA build but GPU not
         currently accessible (driver issue etc). Still use this version
         to match the existing build rather than picking the wrong tag.
      3. nvidia-smi / NVML — for fresh environments without torch.
    """
    # 1 & 2: check existing torch
    r = run_py(
        "import torch; "
        "print(torch.version.cuda or 'None'); "
        "print(torch.cuda.is_available())"
    )
    if r.returncode == 0:
        lines = r.stdout.strip().splitlines()
        if len(lines) >= 2:
            cuda_ver     = lines[0].strip()
            cuda_working = lines[1].strip() == "True"
            if cuda_ver not in ("None", ""):
                status = "working" if cuda_working else "build present but CUDA not available"
                print(f"{TAG}   torch.version.cuda = {cuda_ver}  ({status})")
                return cuda_ver  # always trust this over nvidia-smi

    # 3: nvidia-smi (PATH, then full path)
    smi_candidates = ["nvidia-smi"]
    if IS_WINDOWS:
        smi_candidates.append(r"C:\Windows\System32\nvidia-smi.exe")
    for smi in smi_candidates:
        try:
            r2 = subprocess.run(smi, capture_output=True, text=True, timeout=10)
            if r2.returncode == 0:
                import re
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r2.stdout)
                if m:
                    print(f"{TAG}   nvidia-smi: CUDA {m.group(1)} (driver ceiling)")
                    return m.group(1)
        except Exception:
            pass

    # 4: NVML via ctypes (Windows, no PATH needed)
    if IS_WINDOWS:
        r3 = run_py(
            "import ctypes, sys\n"
            "try:\n"
            "    nvml=ctypes.WinDLL('nvml.dll')\n"
            "    nvml.nvmlInit_v2()\n"
            "    v=ctypes.c_int()\n"
            "    nvml.nvmlSystemGetCudaDriverVersion(ctypes.byref(v))\n"
            "    print(str(v.value//1000)+'.'+str((v.value%1000)//10))\n"
            "    nvml.nvmlShutdown()\n"
            "except Exception as e:\n"
            "    sys.exit(1)\n"
        )
        if r3.returncode == 0 and r3.stdout.strip():
            return r3.stdout.strip()

    return None


def cuda_to_tag(cuda_version: str) -> Optional[str]:
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
# Step 2 — ensure PyTorch CUDA is installed AND working
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


def ensure_torch(cuda_tag: str) -> bool:
    """
    Ensure torch is installed with a working CUDA build.

    Checks BOTH torch.version.cuda AND torch.cuda.is_available(). A broken
    or CPU-only torch may still have a non-None version.cuda string (from a
    previous build that was partially overwritten), so we must verify that
    CUDA actually works before declaring it healthy.
    """
    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    state = get_torch_state()

    if state:
        if state["cuda_available"]:
            print(
                f"{TAG}   torch {state['version']} / CUDA {state['cuda_version']} "
                f"— available and working. No reinstall needed."
            )
            return True

        # torch is installed but CUDA is not working — needs reinstall
        if state["cuda_version"] == "None":
            reason = "CPU-only build (whisperx likely overwrote the CUDA build)"
        else:
            reason = f"CUDA {state['cuda_version']} build present but torch.cuda.is_available() = False"
        print(f"{TAG}   torch installed but broken: {reason}")
        print(f"{TAG}   Reinstalling torch + torchaudio ({cuda_tag}) ...")
    else:
        print(f"{TAG}   torch not installed. Installing ({cuda_tag}) ...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-user", "--force-reinstall",
         "--index-url", index_url,
         "torch", "torchaudio"],
        env=clean_env(),
    )
    if result.returncode != 0:
        return False

    # Verify after install
    final = get_torch_state()
    if final and final["cuda_available"]:
        print(f"{TAG}   torch {final['version']} / CUDA {final['cuda_version']} — OK")
        return True

    print(f"{TAG}   torch reinstalled but CUDA still not available.")
    if final:
        print(f"{TAG}   version={final['version']}  cuda={final['cuda_version']}")
    return False


# ---------------------------------------------------------------------------
# Step 3 — faster-whisper + whisperx (with torch protection)
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
            print(f"{TAG}   torch CUDA overwritten — restoring ...")
            tag = cuda_to_tag(torch_before["cuda_version"])
            if tag:
                base = torch_before["version"].split("+")[0]
                subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "--no-user", "--force-reinstall",
                     "--index-url", f"https://download.pytorch.org/whl/{tag}",
                     f"torch=={base}", f"torchaudio=={base}"],
                    env=clean_env(),
                )
                final = get_torch_state()
                if final and final["cuda_available"]:
                    print(f"{TAG}   torch restored ({final['version']} / CUDA {final['cuda_version']})")
                else:
                    print(f"{TAG}   ERROR: could not restore torch automatically.")
        else:
            print(f"{TAG}   torch CUDA intact after whisperx — OK")


# ---------------------------------------------------------------------------
# Step 4 — cuDNN (Windows only)
# ---------------------------------------------------------------------------

def setup_cudnn(cuda_tag: str) -> None:
    """
    Install nvidia-cudnn-cu12 (or cu11) and copy all .dll files from the
    installed package into the ctranslate2 directory.

    Scans the entire nvidia/cudnn tree for .dll files rather than hardcoding
    specific names, since the exact DLLs present can vary between package
    versions and builds.
    """
    if not IS_WINDOWS:
        print(f"{TAG}   Non-Windows — cuDNN step skipped.")
        return

    # Find ctranslate2 dir
    r = run_py("import ctranslate2, os; print(os.path.dirname(ctranslate2.__file__))")
    if r.returncode != 0:
        print(f"{TAG}   ctranslate2 not installed yet — cuDNN setup deferred.")
        return
    ct2_dir = r.stdout.strip()

    # Check if key DLL already present
    if os.path.exists(os.path.join(ct2_dir, "cudnn64_8.dll")):
        print(f"{TAG}   cuDNN DLLs already present in ctranslate2/ — OK")
        return

    pkg = "nvidia-cudnn-cu11==8.9.7.29" if cuda_tag == "cu118" else "nvidia-cudnn-cu12==8.9.7.29"
    print(f"{TAG}   Installing {pkg} ...")
    run_pip("install", pkg)

    # Find site-packages
    r2 = run_py("import site; print(site.getsitepackages()[0])")
    if r2.returncode != 0:
        print(f"{TAG}   Could not locate site-packages — skipping cuDNN copy.")
        return
    site_pkgs = r2.stdout.strip()

    # Scan the entire nvidia/cudnn tree for all .dll files
    cudnn_root = os.path.join(site_pkgs, "nvidia", "cudnn")
    if not os.path.exists(cudnn_root):
        print(f"{TAG}   cuDNN package directory not found at: {cudnn_root}")
        print(f"{TAG}   Manual fix: copy cuDNN v8 .dll files to {ct2_dir}")
        return

    copied  = 0
    skipped = 0
    for root, dirs, files in os.walk(cudnn_root):
        for filename in files:
            if not filename.lower().endswith(".dll"):
                continue
            src = os.path.join(root, filename)
            dst = os.path.join(ct2_dir, filename)
            if os.path.exists(dst):
                skipped += 1
                continue
            try:
                import shutil
                shutil.copy2(src, dst)
                print(f"{TAG}   Copied {filename}")
                copied += 1
            except Exception as e:
                print(f"{TAG}   Could not copy {filename}: {e}")

    if copied == 0 and skipped == 0:
        print(f"{TAG}   WARNING: No .dll files found in {cudnn_root}")
        print(f"{TAG}   Manual fix: copy cuDNN v8 .dll files to {ct2_dir}")
    else:
        print(f"{TAG}   cuDNN: {copied} copied, {skipped} already present.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILENAME = "deep-whisper-setup.log"


def log_path() -> str:
    """
    Return the path for the setup log file.
    Written next to this module so it stays with the package installation
    and is easy to find for diagnostics.
    """
    return os.path.join(os.path.dirname(__file__), LOG_FILENAME)


def write_log(entries: dict) -> None:
    """
    Append a timestamped entry to the setup log file.

    Args:
        entries: Dict of key/value pairs to record.
    """
    import datetime
    path = log_path()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n[{timestamp}]"]
    for k, v in entries.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"{TAG}   Log written to: {path}")
    except Exception as e:
        print(f"{TAG}   WARNING: could not write log: {e}")


def read_logged_cuda_tag() -> Optional[str]:
    """
    Read the most recently recorded cuda_tag from the log file.

    The log is written on every run so this represents the last known good
    (or intended) CUDA tag for this environment. The user can edit the log
    to change the tag, or delete it to force fresh detection.

    Returns:
        A CUDA tag string (e.g. "cu128") if found, otherwise None.
    """
    path = log_path()
    if not os.path.exists(path):
        return None

    last_tag: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                # Lines are written as "  cuda_tag: cu128"
                if stripped.startswith("cuda_tag:"):
                    value = stripped.split(":", 1)[1].strip()
                    if value.startswith("cu"):
                        last_tag = value
    except Exception:
        return None

    return last_tag


# ---------------------------------------------------------------------------
# Step 5 — verification
# ---------------------------------------------------------------------------

def verify() -> bool:
    checks = [
        ("torch + CUDA",
         "import torch; assert torch.cuda.is_available(), "
         f"f'CUDA not available — torch.version.cuda={{torch.version.cuda}}, "
         f"is_available={{torch.cuda.is_available()}}'"),
        ("faster-whisper",
         "import faster_whisper"),
        ("whisperx",
         "import whisperx"),
        ("silero-vad",
         "from silero_vad import load_silero_vad"),
        ("librosa",
         "import librosa"),
    ]
    all_ok = True
    for label, code in checks:
        r = run_py(code)
        if r.returncode == 0:
            print(f"  OK    {label}")
        else:
            err_text  = (r.stderr + r.stdout).strip()
            err_lines = err_text.splitlines()
            last_err  = err_lines[-1] if err_lines else "unknown error"
            print(f"  FAIL  {label}")
            print(f"        {last_err}")

            # Known issue: torchaudio version incompatibility with whisperx.
            # torchaudio.AudioMetaData was moved between versions. whisperx
            # references the old location on some torchaudio builds.
            # Re-running setup_gpu typically resolves this after torch is
            # correctly installed, as it triggers a whisperx reinstall.
            if label == "whisperx" and "AudioMetaData" in err_text:
                print()
                print(f"        This is a known torchaudio/whisperx compatibility issue.")
                print(f"        Fix: re-run this script — whisperx will be reinstalled")
                print(f"        against the correct torchaudio version.")
                print(f"        If it persists:")
                print(f"          pip install --upgrade --no-user whisperx")

            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

FALLBACK_CUDA_TAG = "cu128"   # default for fresh installs with no torch present


def main() -> int:
    print()
    print("=" * 58)
    print("  deep-whisper GPU setup")
    print("=" * 58)
    print()

    # 0. Pre-flight
    print(f"[0/5] Checking for known dependency conflicts ...")
    fix_pytorch_lightning()

    # 1. Detect CUDA and capture initial torch state
    #    The torch state is logged NOW, before anything is installed,
    #    so there is always a record of the starting conditions.
    print(f"\n[1/5] Detecting CUDA version ...")

    torch_initial = get_torch_state()
    cuda_version  = detect_cuda()
    logged_tag    = read_logged_cuda_tag()

    if logged_tag:
        print(f"{TAG}   Log file: last recorded cuda_tag = {logged_tag}")
        print(f"{TAG}   Log: {log_path()}")

    # Build the log entry from the initial state
    log_entry: dict = {
        "python":            sys.executable,
        "platform":          platform.platform(),
    }
    if torch_initial:
        log_entry["torch_version_before"] = torch_initial["version"]
        log_entry["torch_cuda_before"]    = torch_initial["cuda_version"]
        log_entry["cuda_available_before"]= torch_initial["cuda_available"]
    else:
        log_entry["torch_version_before"] = "not installed"
        log_entry["torch_cuda_before"]    = "n/a"
        log_entry["cuda_available_before"]= False
    log_entry["detected_cuda_version"] = cuda_version or "none"
    log_entry["logged_cuda_tag"]       = logged_tag or "none"

    # Determine the CUDA tag to use.
    #
    # Priority order:
    #   1. torch.version.cuda / nvidia-smi (live detection) — most reliable
    #      when the environment is healthy, always reflects current state.
    #   2. Log file (last recorded tag) — used when live detection fails but
    #      the environment has been set up successfully before. The user can
    #      edit cuda_tag in the log to override, or delete the log entirely
    #      to force fresh detection and fall through to the default.
    #   3. FALLBACK_CUDA_TAG (cu128) — for genuinely fresh environments where
    #      neither torch nor a log exists yet.
    #   4. Interactive prompt — only when torch IS installed but detection
    #      failed AND there is no log to fall back on (unusual edge case).

    if cuda_version is not None:
        cuda_tag = cuda_to_tag(cuda_version)
        if cuda_tag is None:
            print(f"{TAG} CUDA {cuda_version} is below the minimum supported version (11.8).")
            print(f"{TAG} Please update your NVIDIA drivers.")
            return 1
        print(f"{TAG}   Detected: {cuda_tag}  (from environment)")

    elif logged_tag is not None:
        cuda_tag = logged_tag
        print(
            f"{TAG}   Using cuda_tag from log: {cuda_tag}\n"
            f"{TAG}   To use a different version, edit cuda_tag in:\n"
            f"{TAG}     {log_path()}\n"
            f"{TAG}   Or delete the log file for fresh detection."
        )

    elif torch_initial is None:
        # No torch, no log, no detectable CUDA — genuinely fresh environment.
        # Default silently rather than prompting.
        cuda_tag = FALLBACK_CUDA_TAG
        print(
            f"{TAG}   No prior install found. Defaulting to {FALLBACK_CUDA_TAG}.\n"
            f"{TAG}   If this is wrong, re-run with:\n"
            f"{TAG}     python -m deep_whisper.pipeline.setup_gpu  (then edit the log and re-run)"
        )

    else:
        # torch present, no log, detection failed — ask rather than guess.
        print(f"\n{TAG} Could not detect CUDA version automatically.")
        print(f"{TAG} To check manually:")
        print(f"{TAG}   python -c \"import torch; print(torch.version.cuda)\"")
        print(f"{TAG}   -- or run: nvidia-smi")
        print()
        tag_input = input("  Enter your CUDA tag (e.g. cu128, cu121): ").strip()
        if not tag_input.startswith("cu"):
            print("Invalid tag — must start with 'cu' e.g. cu128")
            return 1
        cuda_tag = tag_input

    log_entry["cuda_tag"] = cuda_tag

    # Write initial state to log before touching anything
    print(f"{TAG}   Recording initial state ...")
    write_log(log_entry)

    # 2. Ensure torch CUDA
    print(f"\n[2/5] Ensuring PyTorch CUDA ({cuda_tag}) ...")
    if not ensure_torch(cuda_tag):
        print(f"{TAG} PyTorch installation failed.")
        print(f"{TAG} Try running manually:")
        print(f"{TAG}   pip install --force-reinstall --no-user "
              f"--index-url https://download.pytorch.org/whl/{cuda_tag} "
              f"torch torchaudio")
        return 1

    # 3. GPU deps — capture fresh torch state after step 2 for the
    #    whisperx overwrite protection (not the initial state, since
    #    step 2 may have just installed/repaired torch)
    print(f"\n[3/5] Installing GPU pipeline dependencies ...")
    torch_before_whisperx = get_torch_state()
    install_gpu_deps(torch_before_whisperx)

    # 4. cuDNN
    print(f"\n[4/5] Setting up cuDNN (Windows only) ...")
    setup_cudnn(cuda_tag)

    # 5. Verify and log final state
    print(f"\n[5/5] Verifying installation ...")
    ok = verify()

    torch_final = get_torch_state()
    write_log({
        "outcome":             "success" if ok else "failed",
        "cuda_tag_used":       cuda_tag,
        "torch_version_after": torch_final["version"]        if torch_final else "n/a",
        "torch_cuda_after":    torch_final["cuda_version"]   if torch_final else "n/a",
        "cuda_available_after":torch_final["cuda_available"] if torch_final else False,
    })

    print()
    print("=" * 58)
    if ok:
        print("  Setup complete.")
        print("  Run:  python tests/test_env.py  to confirm end-to-end.")
    else:
        print("  Setup completed with errors — see above.")
        print()
        print("  Common fixes:")
        print("    torch + CUDA fails  →  re-run this script")
        print("    whisperx fails      →  pip install --upgrade whisperx")
        print("    cuDNN missing       →  manually copy cuDNN v8 .dlls")
        print("                            see README §Installation")
    print("=" * 58)
    print()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# deep-whisper · install.ps1
# ===========================
# Windows PowerShell installation script.
# Detects the CUDA version in use by this Python environment and installs
# PyTorch with the correct index URL, then installs all remaining
# dependencies in the correct order.
#
# Usage (from the package root, in a PowerShell terminal):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install.ps1
#
# Optional: pass a Python path if not using the system default
#   .\install.ps1 -PythonExe "C:\Python311\python.exe"
#
# Optional: force a specific CUDA tag (skip auto-detection)
#   .\install.ps1 -CudaTag cu124

param(
    [string]$PythonExe = "python",
    [string]$CudaTag   = ""       # e.g. "cu118", "cu121", "cu124", "cu126", "cu128"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== deep-whisper installer ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Step 0 - verify Python
# ---------------------------------------------------------------------------
Write-Host "[0/6] Checking Python..." -ForegroundColor Yellow
try {
    $pyVer = & $PythonExe --version 2>&1
    Write-Host "      $pyVer" -ForegroundColor Green
} catch {
    Write-Error "Python not found at '$PythonExe'. Pass -PythonExe to specify a path."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 1 - detect the CUDA version in use by this Python environment
# ---------------------------------------------------------------------------
# The right question is not "what is the highest CUDA version this GPU
# supports?" but "what CUDA version is this Python environment already using?"
# These can differ - ComfyUI may be pinned to cu121 on a cu128-capable card.
#
# Detection order:
#   1. torch.version.cuda     - environment-aware, platform-agnostic.
#                               If torch is already installed (always true in
#                               ComfyUI), this is the definitive answer.
#   2. nvidia-smi via PATH    - driver ceiling; only reached on a fresh
#   3. nvidia-smi full path     install where torch is not yet present.
#   4. Python NVML via ctypes - same but no PATH dependency.
#   5. Interactive prompt     - last resort.

Write-Host "[1/6] Detecting CUDA version for this Python environment..." -ForegroundColor Yellow

if ($CudaTag -ne "") {
    Write-Host "      CudaTag override: $CudaTag" -ForegroundColor Green
} else {
    $cudaDetected = $false
    $cudaMajor    = 0
    $cudaMinor    = 0

    # Attempt 1: torch.version.cuda
    # torch.version.cuda is the CUDA version the installed PyTorch wheel was
    # compiled against - exactly what any other CUDA package must match.
    # Returns None for CPU-only builds; ImportError if torch not installed.
    if (-not $cudaDetected) {
        try {
            $torchOut = & $PythonExe -c "import torch; print(torch.version.cuda or 'cpu')" 2>&1
            if ($LASTEXITCODE -eq 0) {
                if ($torchOut.Trim() -eq "cpu") {
                    Write-Host "      torch installed but CPU-only - falling back to hardware detection." -ForegroundColor Yellow
                } elseif ($torchOut -match "^(\d+)\.(\d+)") {
                    $cudaMajor    = [int]$Matches[1]
                    $cudaMinor    = [int]$Matches[2]
                    $cudaDetected = $true
                    Write-Host "      torch.version.cuda: $cudaMajor.$cudaMinor  (environment match)" -ForegroundColor Green
                }
            }
        } catch {}
    }

    # Attempt 2: nvidia-smi via PATH
    if (-not $cudaDetected) {
        try {
            $smiOut = & nvidia-smi 2>&1
            if ($LASTEXITCODE -eq 0) {
                $m = [regex]::Match($smiOut, "CUDA Version:\s*(\d+)\.(\d+)")
                if ($m.Success) {
                    $cudaMajor    = [int]$m.Groups[1].Value
                    $cudaMinor    = [int]$m.Groups[2].Value
                    $cudaDetected = $true
                    Write-Host "      nvidia-smi (PATH): CUDA $cudaMajor.$cudaMinor  (driver ceiling)" -ForegroundColor Green
                }
            }
        } catch {}
    }

    # Attempt 3: nvidia-smi via full Windows path
    if (-not $cudaDetected) {
        $smiPath = "C:\Windows\System32\nvidia-smi.exe"
        if (Test-Path $smiPath) {
            try {
                $smiOut = & $smiPath 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $m = [regex]::Match($smiOut, "CUDA Version:\s*(\d+)\.(\d+)")
                    if ($m.Success) {
                        $cudaMajor    = [int]$m.Groups[1].Value
                        $cudaMinor    = [int]$m.Groups[2].Value
                        $cudaDetected = $true
                        Write-Host "      nvidia-smi (full path): CUDA $cudaMajor.$cudaMinor  (driver ceiling)" -ForegroundColor Green
                    }
                }
            } catch {}
        }
    }

    # Attempt 4: Python ctypes -> NVML
    # nvml.dll ships with every NVIDIA driver. Querying it directly avoids
    # any PATH dependency - works inside ComfyUI embedded Python environments.
    if (-not $cudaDetected) {
        $nvmlPy = "import ctypes, sys`n" +
                  "try:`n" +
                  "    nvml = ctypes.WinDLL('nvml.dll')`n" +
                  "    nvml.nvmlInit_v2()`n" +
                  "    ver = ctypes.c_int()`n" +
                  "    nvml.nvmlSystemGetCudaDriverVersion(ctypes.byref(ver))`n" +
                  "    print(str(ver.value // 1000) + '.' + str((ver.value % 1000) // 10))`n" +
                  "    nvml.nvmlShutdown()`n" +
                  "except Exception:`n" +
                  "    sys.exit(1)"
        try {
            $nvmlOut = & $PythonExe -c $nvmlPy 2>&1
            if ($LASTEXITCODE -eq 0 -and $nvmlOut -match "^(\d+)\.(\d+)") {
                $cudaMajor    = [int]$Matches[1]
                $cudaMinor    = [int]$Matches[2]
                $cudaDetected = $true
                Write-Host "      Python NVML: CUDA $cudaMajor.$cudaMinor  (driver ceiling)" -ForegroundColor Green
            }
        } catch {}
    }

    # Attempt 5: ask the user
    if (-not $cudaDetected) {
        Write-Host ""
        Write-Host "  Could not detect CUDA version automatically." -ForegroundColor Red
        Write-Host "  To find your version:" -ForegroundColor Yellow
        Write-Host "    python -c `"import torch; print(torch.version.cuda)`"" -ForegroundColor White
        Write-Host "    -- or run: nvidia-smi  (look for CUDA Version in top-right)" -ForegroundColor White
        Write-Host ""
        Write-Host "  CUDA tag reference:" -ForegroundColor Yellow
        Write-Host "    CUDA 11.8  ->  cu118" -ForegroundColor White
        Write-Host "    CUDA 12.1  ->  cu121" -ForegroundColor White
        Write-Host "    CUDA 12.4  ->  cu124" -ForegroundColor White
        Write-Host "    CUDA 12.6  ->  cu126" -ForegroundColor White
        Write-Host "    CUDA 12.8  ->  cu128" -ForegroundColor White
        Write-Host ""
        $manualTag = Read-Host "  Enter your CUDA tag (e.g. cu128)"
        if ($manualTag -match "^cu\d+$") {
            $CudaTag = $manualTag
        } else {
            Write-Error "Invalid CUDA tag. Must be in the form cu118, cu121, etc."
            exit 1
        }
    }

    # Map detected version to the nearest available PyTorch index tag
    if ($cudaDetected) {
        if     ($cudaMajor -ge 12 -and $cudaMinor -ge 8) { $CudaTag = "cu128" }
        elseif ($cudaMajor -ge 12 -and $cudaMinor -ge 6) { $CudaTag = "cu126" }
        elseif ($cudaMajor -ge 12 -and $cudaMinor -ge 4) { $CudaTag = "cu124" }
        elseif ($cudaMajor -ge 12 -and $cudaMinor -ge 1) { $CudaTag = "cu121" }
        elseif ($cudaMajor -ge 11 -and $cudaMinor -ge 8) { $CudaTag = "cu118" }
        else {
            Write-Error "CUDA $cudaMajor.$cudaMinor is below the minimum supported version (11.8). Please update your NVIDIA drivers."
            exit 1
        }
        Write-Host "      Selected PyTorch tag: $CudaTag" -ForegroundColor Green
    }
}

$torchIndexUrl = "https://download.pytorch.org/whl/$CudaTag"

# ---------------------------------------------------------------------------
# Isolate pip from user site-packages for the rest of this script
# ---------------------------------------------------------------------------
# ComfyUI (and some custom nodes) explicitly add the user site-packages
# directory to sys.path via sys.path.append(). pip inherits this and sees
# any broken packages there during dependency resolution — even with --no-user,
# which only controls install targets, not what pip can see.
# PYTHONNOUSERSITE=1 prevents Python from adding user site-packages to
# sys.path in every subprocess we spawn from here on.
$env:PYTHONNOUSERSITE = "1"
Write-Host "      PYTHONNOUSERSITE=1  (user site-packages isolated from pip resolution)" -ForegroundColor Gray

# ---------------------------------------------------------------------------
# Pre-flight: fix known broken packages before pip resolves anything
# ---------------------------------------------------------------------------
# pytorch-lightning < 2.0.0 ships with a malformed version specifier
# (torch>=1.9.*) that pip 24.1+ rejects during dependency resolution —
# even when --no-user is set, if the package is in the target environment.
# Upgrade it before any install that triggers dependency resolution.
Write-Host "[pre] Checking for known dependency conflicts..." -ForegroundColor Yellow

$plResult = & $PythonExe -m pip show pytorch-lightning 2>&1
if ($LASTEXITCODE -eq 0) {
    $plVersion = ($plResult | Select-String "^Version:").ToString() -replace "Version:\s*", ""
    $plMajor   = [int]($plVersion.Split(".")[0])
    if ($plMajor -lt 2) {
        Write-Host "      Found pytorch-lightning $plVersion (invalid metadata for pip 24.1+)." -ForegroundColor Yellow
        Write-Host "      Upgrading to a version with valid metadata..." -ForegroundColor Yellow
        $upgradeResult = & $PythonExe -m pip install --no-user "pytorch-lightning>=2.0.0" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      pytorch-lightning upgraded." -ForegroundColor Green
        } else {
            Write-Host "      Upgrade failed — removing pytorch-lightning instead..." -ForegroundColor Yellow
            & $PythonExe -m pip uninstall pytorch-lightning -y
            if ($LASTEXITCODE -eq 0) {
                Write-Host "      pytorch-lightning removed." -ForegroundColor Green
            } else {
                Write-Host "      WARNING: Could not remove pytorch-lightning." -ForegroundColor Red
                Write-Host "      If the install fails, run manually:" -ForegroundColor Yellow
                Write-Host "        $PythonExe -m pip uninstall pytorch-lightning -y" -ForegroundColor White
            }
        }
    } else {
        Write-Host "      pytorch-lightning $plVersion — OK." -ForegroundColor Gray
    }
} else {
    Write-Host "      pytorch-lightning not found — OK." -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# Step 2 - install PyTorch + torchaudio + torchvision (CUDA build)
# ---------------------------------------------------------------------------
Write-Host "[2/6] Installing PyTorch + torchaudio + torchvision ($CudaTag)..." -ForegroundColor Yellow
Write-Host "      Index URL: $torchIndexUrl"
& $PythonExe -m pip install --no-user torch torchaudio torchvision --index-url $torchIndexUrl
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyTorch installation failed. Check your internet connection and CUDA tag."
    exit 1
}
Write-Host "      PyTorch installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 3 - install cuDNN 8 and copy DLLs for CTranslate2
# ---------------------------------------------------------------------------
# CTranslate2 4.x (used by faster-whisper) requires cuDNN 8.x on Windows.
# It does NOT bundle cuDNN and cuDNN 9.x will NOT work - CTranslate2 4.x
# looks for the v8 DLL naming convention (cudnn64_8.dll etc).
#
# nvidia-cudnn-cu12 installs the DLLs into site-packages/nvidia/cudnn/bin/
# but CTranslate2 cannot find them there on Windows. We copy the three
# required DLLs directly into the ctranslate2 package directory.
Write-Host "[3/6] Installing cuDNN 8 for CTranslate2..." -ForegroundColor Yellow

$cudnnPackage = if ($CudaTag -eq "cu118") { "nvidia-cudnn-cu11==8.9.7.29" } `
                else                       { "nvidia-cudnn-cu12==8.9.7.29" }

& $PythonExe -m pip install --no-user $cudnnPackage
if ($LASTEXITCODE -ne 0) {
    Write-Error "cuDNN installation failed."
    exit 1
}

$sitePackages = & $PythonExe -c "import site; print(site.getsitepackages()[0])" 2>&1
$cudnnBinDir  = Join-Path $sitePackages "nvidia\cudnn\bin"
$ct2Dir       = & $PythonExe -c "import ctranslate2, os; print(os.path.dirname(ctranslate2.__file__))" 2>&1

$requiredDlls = @(
    "cudnn64_8.dll",
    "cudnn_ops_infer64_8.dll",
    "cudnn_cnn_infer64_8.dll"
)

$copyCount   = 0
$missingDlls = @()

foreach ($dll in $requiredDlls) {
    $src = Join-Path $cudnnBinDir $dll
    $dst = Join-Path $ct2Dir $dll

    if (Test-Path $dst) {
        Write-Host "      $dll already present in ctranslate2 - skipped." -ForegroundColor Gray
        $copyCount++
        continue
    }

    if (Test-Path $src) {
        Copy-Item $src $dst
        Write-Host "      Copied $dll -> ctranslate2/" -ForegroundColor Green
        $copyCount++
    } else {
        $missingDlls += $dll
        Write-Host "      WARNING: $dll not found at $src" -ForegroundColor Yellow
    }
}

if ($missingDlls.Count -gt 0) {
    Write-Host ""
    Write-Host "  Some cuDNN DLLs could not be copied automatically:" -ForegroundColor Yellow
    foreach ($dll in $missingDlls) { Write-Host "    - $dll" -ForegroundColor White }
    Write-Host ""
    Write-Host "  Manual fallback:" -ForegroundColor Yellow
    Write-Host "    1. Download cuDNN v8 from https://developer.nvidia.com/cudnn-downloads" -ForegroundColor White
    Write-Host "    2. Extract the archive" -ForegroundColor White
    Write-Host "    3. Copy bin\*.dll files to: $ct2Dir" -ForegroundColor White
    Write-Host ""
    $cont = Read-Host "  Continue installation anyway? [y/N]"
    if ($cont -ne "y" -and $cont -ne "Y") { exit 1 }
} else {
    Write-Host "      cuDNN DLLs in place ($copyCount/$($requiredDlls.Count))." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Step 4 - verify CUDA is visible to PyTorch
# ---------------------------------------------------------------------------
Write-Host "[4/6] Verifying PyTorch CUDA..." -ForegroundColor Yellow
$cudaCheck = & $PythonExe -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaCheck -ne "True") {
    Write-Host ""
    Write-Host "  WARNING: torch.cuda.is_available() returned: $cudaCheck" -ForegroundColor Red
    Write-Host "  PyTorch was installed but cannot see your GPU." -ForegroundColor Red
    Write-Host "  Possible causes:" -ForegroundColor Yellow
    Write-Host "    - The installed CUDA tag does not match your driver version" -ForegroundColor White
    Write-Host "    - Your NVIDIA drivers are outdated (update via GeForce Experience)" -ForegroundColor White
    Write-Host "    - You are running in a non-GPU environment" -ForegroundColor White
    Write-Host ""
    $cont = Read-Host "  Continue installation anyway? [y/N]"
    if ($cont -ne "y" -and $cont -ne "Y") { exit 1 }
} else {
    $gpuName = & $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    $vram    = & $PythonExe -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1))" 2>&1
    Write-Host "      CUDA OK - $gpuName ($vram GB)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Step 5 - install faster-whisper, then whisperx (with torch protection)
# ---------------------------------------------------------------------------
# whisperx is known to pull in a CPU-only torch as a transitive dependency,
# silently overwriting a CUDA-capable build. We capture the torch state
# before, install, then detect and restore automatically if broken.

Write-Host "[5/6] Installing faster-whisper..." -ForegroundColor Yellow
& $PythonExe -m pip install --no-user "faster-whisper>=1.0.0"
if ($LASTEXITCODE -ne 0) { Write-Error "faster-whisper installation failed."; exit 1 }

# Capture torch state before whisperx
$torchBefore = & $PythonExe -c "import torch; print(torch.__version__); print(torch.version.cuda or 'None'); print(torch.cuda.is_available())" 2>&1
$torchBeforeLines = ($torchBefore -split "`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$torchVersionBefore  = if ($torchBeforeLines.Count -ge 1) { $torchBeforeLines[0] } else { "" }
$torchCudaBefore     = if ($torchBeforeLines.Count -ge 2) { $torchBeforeLines[1] } else { "None" }
$torchCudaOkBefore   = if ($torchBeforeLines.Count -ge 3) { $torchBeforeLines[2] -eq "True" } else { $false }

Write-Host "      torch before whisperx: $torchVersionBefore / CUDA $torchCudaBefore / available=$torchCudaOkBefore" -ForegroundColor Gray

Write-Host "      Installing whisperx..." -ForegroundColor Yellow
& $PythonExe -m pip install --no-user "whisperx>=3.0.0"
if ($LASTEXITCODE -ne 0) { Write-Error "whisperx installation failed."; exit 1 }

# Detect and repair torch breakage caused by whisperx
if ($torchCudaOkBefore) {
    $torchAfter = & $PythonExe -c "import torch; print(torch.__version__); print(torch.version.cuda or 'None'); print(torch.cuda.is_available())" 2>&1
    $torchAfterLines = ($torchAfter -split "`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
    $torchVersionAfter = if ($torchAfterLines.Count -ge 1) { $torchAfterLines[0] } else { "" }
    $torchCudaAfter    = if ($torchAfterLines.Count -ge 2) { $torchAfterLines[1] } else { "None" }
    $torchCudaOkAfter  = if ($torchAfterLines.Count -ge 3) { $torchAfterLines[2] -eq "True" } else { $false }

    $cudaBroken = (-not $torchCudaOkAfter) -or ($torchCudaAfter -ne $torchCudaBefore)

    if ($cudaBroken) {
        Write-Host ""
        Write-Host "  WARNING: whisperx modified torch." -ForegroundColor Yellow
        Write-Host "    Before: $torchVersionBefore / CUDA $torchCudaBefore" -ForegroundColor White
        Write-Host "    After:  $torchVersionAfter / CUDA $torchCudaAfter" -ForegroundColor White
        Write-Host "  Restoring torch CUDA build..." -ForegroundColor Yellow

        # Strip local version suffix (e.g. 2.6.0+cu128 -> 2.6.0) for pip lookup
        $torchBaseVersion = $torchVersionBefore -replace '\+.*$', ''

        & $PythonExe -m pip install `
            --no-user `
            --force-reinstall `
            --index-url $torchIndexUrl `
            "torch==$torchBaseVersion" `
            "torchaudio==$torchBaseVersion" `
            "torchvision==$torchBaseVersion"

        $torchFinal = & $PythonExe -c "import torch; print(torch.cuda.is_available())" 2>&1
        if ($torchFinal.Trim() -eq "True") {
            Write-Host "      torch CUDA restored successfully." -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "  ERROR: Could not restore torch CUDA automatically." -ForegroundColor Red
            Write-Host "  Please run this command manually:" -ForegroundColor Yellow
            Write-Host "    $PythonExe -m pip install --force-reinstall --no-user \" -ForegroundColor White
            Write-Host "      --index-url $torchIndexUrl \" -ForegroundColor White
            Write-Host "      torch==$torchBaseVersion torchaudio==$torchBaseVersion" -ForegroundColor White
        }
    } else {
        Write-Host "      torch CUDA intact after whisperx. ($torchVersionAfter / CUDA $torchCudaAfter)" -ForegroundColor Green
    }
}

Write-Host "      faster-whisper + whisperx installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 6 - install remaining dependencies
# ---------------------------------------------------------------------------
Write-Host "[6/6] Installing remaining dependencies..." -ForegroundColor Yellow
& $PythonExe -m pip install --no-user `
    "silero-vad>=5.0.0" `
    "librosa>=0.10.0" `
    "soundfile>=0.12.0" `
    "num2words>=0.5.13" `
    "diff-match-patch>=20230430" `
    "numpy>=1.24,<2.3"
if ($LASTEXITCODE -ne 0) { Write-Error "Dependency installation failed."; exit 1 }
Write-Host "      All dependencies installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Verifying installation ===" -ForegroundColor Cyan

# Build the verification script as a plain string (no here-string) to avoid
# any PowerShell parser ambiguity with quotes inside the Python code.
$verifyLines = @(
    "import sys",
    "checks = [",
    "    ('torch',            lambda: __import__('torch').__version__),",
    "    ('torch.cuda',       lambda: str(__import__('torch').cuda.is_available())),",
    "    ('torchaudio',       lambda: __import__('torchaudio').__version__),",
    "    ('torchvision',      lambda: __import__('torchvision').__version__),",
    "    ('faster_whisper',   lambda: __import__('faster_whisper').__version__),",
    "    ('whisperx',         lambda: 'OK' if __import__('whisperx') else ''),",
    "    ('silero_vad',       lambda: 'OK' if __import__('silero_vad') else ''),",
    "    ('librosa',          lambda: __import__('librosa').__version__),",
    "    ('soundfile',        lambda: __import__('soundfile').__version__),",
    "    ('num2words',        lambda: __import__('num2words').__version__),",
    "    ('diff_match_patch', lambda: 'OK' if __import__('diff_match_patch') else ''),",
    "]",
    "all_ok = True",
    "for name, fn in checks:",
    "    try:",
    "        val = fn()",
    "        print('  OK  ' + name.ljust(20) + ' ' + str(val))",
    "    except Exception as e:",
    "        print('  !!  ' + name.ljust(20) + ' FAILED: ' + str(e))",
    "        all_ok = False",
    "sys.exit(0 if all_ok else 1)"
)
$verifyScript = $verifyLines -join "`n"

& $PythonExe -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  One or more packages failed verification." -ForegroundColor Red
    Write-Host "  Check the output above and re-run the failing step manually." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Installation complete. Run test_env.py to confirm end-to-end." -ForegroundColor Green
Write-Host ""

# Restore environment
Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue

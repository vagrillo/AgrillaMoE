# AgrillaMoE — build Windows (staticamente linkato, backend CUDA incorporato)
#
# Parametri opzionali:
#   -CudaArch  architetture CUDA (default: native; es. "61", "75;86")
#   -BuildDir  cartella di build (default: D:\agrilla-build; cade su C:\agrilla-build se D: non c'e')
#
# Requisiti: Visual Studio 2019 BuildTools (VC++ + CMake+Ninja bundled) e CUDA toolkit 12+.
# Da eseguire in Windows PowerShell:  powershell -ExecutionPolicy Bypass -File build-windows.ps1
param(
    [string]$CudaArch = "native",
    [string]$BuildDir = "",
    [int]$Native = 1,
    [int]$Jobs = 4,
    [int]$Vulkan = 0,
    [int]$Hip = 0,
    [string]$AmdTargets = "gfx1101;gfx1100"
)

$ErrorActionPreference = "Stop"

$VS     = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools"
if (-not (Test-Path "$VS\VC\Auxiliary\Build\vcvars64.bat")) {
    # fallback: proviamo le altre edition/anni di VS
    $cand = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio" -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
            Where-Object { Test-Path "$($_.FullName)\VC\Auxiliary\Build\vcvars64.bat" } |
            Sort-Object FullName -Descending
    if ($cand) { $VS = $cand[0].FullName } else { throw "Visual Studio (BuildTools) con vcvars64.bat non trovato" }
}
# CMake: serve >= 3.24 per una detection CUDA decente; il bundle VS2019 e' 3.20.
# Ordine: copia portabile su D:\tools\cmake, poi PATH, poi bundle VS.
$CMakeExe = $null
if (Test-Path "D:\tools\cmake\bin\cmake.exe") { $CMakeExe = "D:\tools\cmake\bin\cmake.exe" }
if (-not $CMakeExe) { $CMakeExe = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source }
if (-not $CMakeExe) {
    $CMakeExe = Get-ChildItem "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty FullName
}
if (-not $CMakeExe) { throw "cmake non trovato (D:\tools\cmake, PATH, o bundle VS)" }
Write-Host "uso cmake:   $CMakeExe ($(& $CMakeExe --version | Select-Object -First 1))"
$NinjaDir = "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path "$NinjaDir\ninja.exe") { $env:PATH = "$NinjaDir;$env:PATH" }

# CUDA toolkit 12.x: rilevato dal nvcc presente su PATH, oppure cartelle note
# (installazioni non standard tipo D:\cudatooliki12.6 funzionano automaticamente)
$NvccPath = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
if (-not $NvccPath) {
    $cand = @(
        "D:\cudatooliki12.6",
        "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        "D:\CUDA", "C:\CUDA"
    ) | Where-Object { Test-Path "$_\bin\nvcc.exe" } | Select-Object -First 1
    if ($cand) { $NvccPath = "$cand\bin\nvcc.exe" }
}
if (-not $NvccPath) { throw "nvcc.exe non trovato (PATH o cartelle note)" }
$CudaRoot = Split-Path (Split-Path $NvccPath)
$env:PATH = "$CudaRoot\bin;$env:PATH"
Write-Host "uso VS:      $VS"
Write-Host "uso cmake:   $CMakeExe ($(& $CMakeExe --version | Select-Object -First 1))"
Write-Host "uso CUDA:    $CudaRoot ($(& "$CudaRoot\bin\nvcc.exe" --version | Select-Object -Last 1))"

if (-not $BuildDir) {
    if (Test-Path "D:\") { $BuildDir = "D:\agrilla-build" } else { $BuildDir = "C:\agrilla-build" }
}

$Src = $PSScriptRoot

# ---- backend aggiuntivi: Vulkan (NVIDIA+AMD+Intel) e HIP/ROCm (AMD) ----
$ExtraDefs = ""
if ($Vulkan -eq 1) {
    $Vsdk = $env:VULKAN_SDK
    if (-not $Vsdk) {
        # layout con versione (C:\VulkanSDK\1.3.xxx) oppure flat (D:\VulkanSDK)
        if (Test-Path "D:\VulkanSDK\Bin\glslc.exe") {
            $Vsdk = "D:\VulkanSDK"
        } else {
            $Vsdk = Get-ChildItem "C:\VulkanSDK","D:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
                    Where-Object { Test-Path "$($_.FullName)\Bin\glslc.exe" } |
                    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
        }
    }
    if ($Vsdk) {
        $env:VULKAN_SDK = $Vsdk
        Write-Host "uso Vulkan SDK: $Vsdk"
        $ExtraDefs += " -DGGML_VULKAN=ON"
    } else {
        Write-Warning "Vulkan=1 ma nessun Vulkan SDK trovato (VULKAN_SDK o C:\VulkanSDK): backend Vulkan saltato"
    }
}
if ($Hip -eq 1) {
    # HIP e' incompatibile con CUDA nello stesso binario
    $ExtraDefs += " -DGGML_CUDA=OFF -DGGML_HIP=ON -DAMDGPU_TARGETS=$AmdTargets"
    if (-not $env:HIP_PATH) {
        $HipDir = Get-ChildItem "C:\Program Files\AMD\ROCm\*" -Directory -ErrorAction SilentlyContinue |
                  Sort-Object Name -Descending | Select-Object -First 1
        if ($HipDir) { $env:HIP_PATH = $HipDir.FullName }
    }
    Write-Host "HIP/ROCm ON (target: $AmdTargets)$(if ($env:HIP_PATH) { ", HIP_PATH=$($env:HIP_PATH)" } else { ' (HIP_PATH non rilevato: installare AMD HIP SDK)' })"
}

cmd /c "`"$VS\VC\Auxiliary\Build\vcvars64.bat`" && `"$CMakeExe`" -S `"$Src`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=$Native$ExtraDefs -DCMAKE_CUDA_COMPILER=`"$NvccPath`" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES=$CudaArch && `"$CMakeExe`" --build `"$BuildDir`" --target agrillamoe --parallel $Jobs"
if ($LASTEXITCODE -ne 0) { throw "build fallita (codice $LASTEXITCODE)" }

New-Item -ItemType Directory -Force -Path "$Src\dist\windows" | Out-Null
$Exe = "$BuildDir\agrillamoe.exe"
if (-not (Test-Path $Exe)) { $Exe = "$BuildDir\bin\agrillamoe.exe" }
Copy-Item -Force $Exe "$Src\dist\windows\agrillamoe.exe"
Write-Host ""
Write-Host "OK: $Src\dist\windows\agrillamoe.exe"

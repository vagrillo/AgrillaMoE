# diag-gpu.ps1 — diagnostica: perche' la GPU non viene usata partendo da D:\tmp
$ErrorActionPreference = "Continue"
$exe = "D:\tmp\agrillamoe-vulkan.exe"
$model = "D:\models\Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
$log = "$env:TEMP\agrilla-diag.log"

Write-Host "=== 1) PATH visibile al processo (cartelle CUDA?) ==="
($env:Path -split ';' | Where-Object { $_ -match 'cuda|CUDA' }) | ForEach-Object { Write-Host "  $_" }

Write-Host "`n=== 2) dispositivi visti dal binario ==="
& $exe --list-devices --no-browser 2>&1 | Select-String -Pattern "CUDA|Vulkan|GPU|Backend|backend|device" | Select-Object -First 15 | ForEach-Object { $_.Line }

Write-Host "`n=== 3) avvio di default da D:\tmp ==="
Set-Location D:\tmp
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $exe -WorkingDirectory "D:\tmp" -ArgumentList @(
    "-m",$model,"--host","127.0.0.1","--port","8086","--no-browser","--reasoning","off"
) -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\agrilla-diag.out" -PassThru

Start-Sleep -Seconds 25
Write-Host "--- prime righe utili del log (backend/init) ---"
Get-Content $log | Select-String -Pattern "ggml|cuda|vulkan|CUDA|Vulkan|device|offload|n_gpu|load" | Select-Object -First 20 | ForEach-Object { $_.Line }

Write-Host "`n--- VRAM / utilizzo GPU durante il caricamento ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# attende ready o morte
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8086/health" -TimeoutSec 3
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch {}
    if ($p.HasExited) { break }
}
Write-Host "`nready=$ok (uscito=$($p.HasExited))"
Write-Host "--- righe offload/KV dopo il load ---"
Get-Content $log | Select-String -Pattern "offloaded|KV|buffer|CUDA0|Vulkan0|layers" | Select-Object -First 12 | ForEach-Object { $_.Line }
Write-Host "--- VRAM dopo il load ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

if ($ok) {
    Write-Host "`n--- mini generazione ---"
    $body = '{"messages":[{"role":"user","content":"Scrivi 5 parole."}],"max_tokens":30}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8086/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600
    $sw.Stop()
    $n = $r.usage.completion_tokens
    Write-Host "$n token in $([math]::Round($sw.Elapsed.TotalSeconds,1))s = $([math]::Round($n/$sw.Elapsed.TotalSeconds,2)) tok/s"
    Write-Host "--- GPU durante la generazione ---"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Write-Host "`nfine diagnostica."

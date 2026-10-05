$ErrorActionPreference = "Continue"
$exe = "C:\AI\src\llamacpp\AgrillaMoE\dist\windows\agrillamoe.exe"
$model = "D:\models\Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
$log = "$env:TEMP\gpu-stream.log"
Remove-Item $log, "$env:TEMP\gpu-stream.out" -ErrorAction SilentlyContinue
Write-Host "--- avvio gpu-streaming (--agrilla-gpu-streaming, KV q4_0) ---"
$p = Start-Process -FilePath $exe -WorkingDirectory "D:\tmp" -ArgumentList @(
    "--agrilla-gpu-streaming","-m",$model,
    "--host","127.0.0.1","--port","8087","--no-browser",
    "--reasoning","off","-c","4096","-np","1","-ctk","q4_0","-ctv","q4_0"
) -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\gpu-stream.out" -PassThru
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 3
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:8087/health" -TimeoutSec 3; if ($h.status -eq "ok") { $ok = $true; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { Write-Host "SERVER NON PRONTO (uscito=$($p.HasExited))"; Get-Content $log -Tail 12 -ErrorAction SilentlyContinue; exit 1 }
Write-Host "server pronto."
Write-Host "--- VRAM durante idle dopo load ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
$body = '{"messages":[{"role":"user","content":"Scrivi un paragrafo lungo e dettagliato sull universo."}],"max_tokens":120,"temperature":0}'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri "http://127.0.0.1:8087/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 900
$sw.Stop()
$n = $r.usage.completion_tokens
Write-Host "generati $n token in $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
Start-Sleep 2
Write-Host "--- VRAM/util DOPO la generazione ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
Get-Content $log | Select-String "eval time|prompt eval|CUDA error|gpu-streaming" | Select-Object -Last 5 | ForEach-Object { $_.Line }
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue

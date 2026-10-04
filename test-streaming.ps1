# test-streaming.ps1 — verifica della modalita' streaming di AgrillaMoE su Windows
# uso: powershell -ExecutionPolicy Bypass -File test-streaming.ps1
$ErrorActionPreference = "Continue"
$exe = "$PSScriptRoot\dist\windows\agrillamoe.exe"
$model = "D:\models\Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
$log = "$env:TEMP\agrilla-streaming-test.log"

if (-not (Test-Path $model)) { Write-Host "MODELLO MANCANTE: $model"; exit 1 }

Write-Host "avvio AgrillaMoE streaming su :8084..."
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $exe -ArgumentList @(
    "--agrilla-streaming","-m",$model,
    "--host","127.0.0.1","--port","8084","--no-browser",
    "--reasoning","off","-c","4096","-np","1"
) -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\agrilla-streaming-test.out" -PassThru

# attesa health fino a 6 minuti (primo load da SSD di 9.4 GB)
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8084/health" -TimeoutSec 3
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    Write-Host "SERVER NON PRONTO (exit=$($p.HasExited))"
    Get-Content $log -Tail 25
    exit 1
}
Write-Host "server pronto."

Write-Host "`n--- VRAM dopo il load ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

Write-Host "`n--- riepilogo AgrillaMoE ---"
Select-String -Path $log -Pattern "AgrillaMoE\]" | Select-Object -First 8 | ForEach-Object { $_.Line }

Write-Host "`n--- generazione (tempo misurato) ---"
$body = '{"messages":[{"role":"user","content":"Scrivi una frase di 15 parole sull universo."}],"max_tokens":60}'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri "http://127.0.0.1:8084/v1/chat/completions" -Method Post `
     -ContentType "application/json" -Body $body -TimeoutSec 600
$sw.Stop()
$txt = $r.choices[0].message.content
$n = $r.usage.completion_tokens
Write-Host "risposta ($n token in $([math]::Round($sw.Elapsed.TotalSeconds,1)) s = $([math]::Round($n/$sw.Elapsed.TotalSeconds,2)) tok/s):"
Write-Host $txt

Write-Host "`n--- timing dal log ---"
Get-Content $log -Tail 6 | Select-String "eval time|prompt eval"

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Write-Host "`nfine test."

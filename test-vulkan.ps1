# test-vulkan.ps1 — verifica del backend Vulkan di AgrillaMoE (stesso percorso
# di codice che usera' una Radeon RX 7800 XT). Uso:
#   powershell -ExecutionPolicy Bypass -File test-vulkan.ps1 [percorso-exe] [modello]
$ErrorActionPreference = "Continue"
$exe    = if ($args[0]) { $args[0] } else { "$PSScriptRoot\dist\windows\agrillamoe-vulkan.exe" }
$model  = if ($args[1]) { $args[1] } else { "D:\models\Qwen3.6-35B-A3B-UD-IQ1_M.gguf" }
$log    = "$env:TEMP\agrilla-vk-test.log"

if (-not (Test-Path $exe))    { Write-Host "EXE MANCANTE: $exe"; exit 1 }
if (-not (Test-Path $model))  { Write-Host "MODELLO MANCANTE: $model"; exit 1 }

Write-Host "--- dispositivi visibili (attesi CUDA0 + Vulkan0) ---"
& $exe --list-devices --no-browser 2>&1 | Select-Object -First 14

Write-Host "`n--- avvio su GPU VULKAN (:8085) ---"
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $exe -ArgumentList @(
    "-m",$model,"--device","Vulkan0",
    "--host","127.0.0.1","--port","8085","--no-browser",
    "--reasoning","off","-c","4096","-np","1","--no-moe-expansion"
) -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\agrilla-vk.out" -PassThru

$ok = $false
for ($i = 0; $i -lt 80; $i++) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8085/health" -TimeoutSec 3
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    Write-Host "SERVER NON PRONTO (uscito=$($p.HasExited))"; Get-Content $log -Tail 20; exit 1
}
Write-Host "server pronto su Vulkan."

Write-Host "`n--- generazione su Vulkan ---"
$body = '{"messages":[{"role":"user","content":"Quanto fa 7*6? Una riga."}],"max_tokens":40}'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri "http://127.0.0.1:8085/v1/chat/completions" -Method Post `
     -ContentType "application/json" -Body $body -TimeoutSec 600
$sw.Stop()
$n = $r.usage.completion_tokens
Write-Host "risposta ($n token in $([math]::Round($sw.Elapsed.TotalSeconds,1)) s = $([math]::Round($n/$sw.Elapsed.TotalSeconds,2)) tok/s):"
Write-Host $r.choices[0].message.content
Get-Content $log -Tail 4 | Select-String "eval time"
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Write-Host "`nfine test Vulkan."

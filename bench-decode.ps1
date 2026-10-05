# bench-decode.ps1 — confronto velocita' di decodifica (tok/s) su questo PC
$ErrorActionPreference = "Continue"
$model = "D:\models\Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
$body  = '{"messages":[{"role":"user","content":"Scrivi un paragrafo lungo e dettagliato sull universo."}],"max_tokens":220,"temperature":0}'

function Run-Case($nome, $exe, $extra, $port) {
    $log = "$env:TEMP\bench-$port.log"
    Remove-Item $log, "$env:TEMP\bench-$port.out" -ErrorAction SilentlyContinue
    Write-Host "`n=== $nome ==="
    $arglist = @("-m",$model,"--host","127.0.0.1","--port","$port","--no-browser",
                 "--reasoning","off","-c","4096","-np","1") + $extra
    $p = Start-Process -FilePath $exe -ArgumentList $arglist `
        -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\bench-$port.out" -PassThru
    $ok = $false
    for ($i = 0; $i -lt 100; $i++) {
        Start-Sleep -Seconds 3
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
            if ($h.status -eq "ok") { $ok = $true; break }
        } catch {}
        if ($p.HasExited) { break }
    }
    if (-not $ok) { Write-Host "FALLITO"; Get-Content $log -Tail 6 -ErrorAction SilentlyContinue; return }
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post `
         -ContentType "application/json" -Body $body -TimeoutSec 900
    $n = $r.usage.completion_tokens
    $line = (Get-Content $log | Select-String "eval time = " | Select-Object -Last 1).Line
    Write-Host "token generati: $n"
    Write-Host $line
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

$cudaExe = "C:\AI\src\llamacpp\AgrillaMoE\dist\windows\agrillamoe.exe"
$vkExe   = "D:\tmp\agrillamoe-vulkan.exe"

Run-Case "1) streaming su CUDA (default)"    $cudaExe @() 8090
Run-Case "2) streaming su Vulkan (--device)" $vkExe   @("--device","Vulkan0") 8091
Run-Case "3) split classico (no streaming)"  $cudaExe @("--agrilla-no-autostream","-ngl","99") 8092
Write-Host "`nfine benchmark."

@echo off
rem AgrillaMoE Lite - modalita' streaming per GPU con poca VRAM (stile DS4):
rem   - esperti del MoE streamati da disco via mmap (calcolati solo quelli
rem     instradati dal router), attenzione/KV in GPU, prefetch dei layer
rem     successivi in RAM
rem   - modello consigliato su D: (o dove preferisci): UD-IQ1_M (~9.4 GB)
rem     hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ1_M.gguf --local-dir D:\models
rem Uso: agrillamoe-lite.cmd [parametri extra passati ad agrillamoe]
rem Se agrillamoe.exe non e' accanto a questo script, mettilo nel PATH.
agrillamoe.exe --agrilla-streaming --agrilla-models-dir D:\models %*

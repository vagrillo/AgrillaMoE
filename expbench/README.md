# MoE-expansion quality benchmark — 5 problems × 10 routing configs

Benchmark di **qualità** (non di velocità): 5 problemi di programmazione media di
natura diversa vengono risolti dal modello con **10 configurazioni diverse di
MoE-expansion**, salvando reasoning e risposta per ognuno. I 50 output passano
poi a un **LLM judge accecato** (candidati etichettati A-J con mescolamento
stabile) che assegna punteggi e decreta la migliore configurazione; in parallelo
un **test oggettivo** valuta ogni soluzione sui test di riferimento.

## Problemi (Python, nature diverse)

| id | natura | titolo |
|---|---|---|
| p1 | programmazione dinamica su sequenze | Cheapest pillar route (salti ≤ k, costo = differenza altezze) |
| p2 | stringhe / sliding window | Longest substring con ≤ k caratteri distinti |
| p3 | grafi / BFS con stato | Griglia con chiave e porta (stato key/no-key) |
| p4 | intervalli / sweep line | Numero minimo di sale riunioni |
| p5 | parsing / stack | Valutatore di espressioni intere con parentesi |

Ogni problema ha 6 test di riferimento eseguiti in subprocess con timeout.

## Le 10 configurazioni

stock top-8 · 12/0,8 · 16/0,8 · 20/0,8 · 24/0,8 (L25-39) · 20/0,6 · 20/0,9
(L25-39) · 20/0,8/L0-39 · 20/0,8/L30-39 · 16/0,7/L20-39. Dettagli in
`configs.json`.

## Parametri di generazione

`--reasoning-budget 24576` (thinking), ctx 32768, KV q8_0, temperatura 0,
max_tokens 28672, un campione per problema.

## Uso (VM GPU 16GB, es. V100, quant UD-Q3_K_XL)

```bash
EXPBENCH_MODEL=models/Q3KXL.gguf bash run-expbench.sh
```

Su 16GB il modello 15,7GB non lascia spazio a KV 32K: lo script usa di default
gli **esperti su CPU** (`-cmoe`, modalità già validata) — per una GPU 24GB+
impostare `EXPBENCH_CM=""` (full GPU, più veloce). Il run è **resumabile**: i
file esistenti in `runs/<cfg>/p<k>.json` vengono saltati.

## Output

- `runs/<cfg>/p<k>.json` — reasoning + risposta + usage per ogni run
- `runs/tests-summary.json` — test_ratio per config/problema (oggettivo)
- `runs/judge-report.json` — punteggi e ranking del judge accecato + mapping
  label→config (pubblicato solo dopo, per trasparenza)
- report finale aggregato: ranking delle configurazioni

## Nota metodologica

Il judge è lo stesso modello (self-judge): utile per confronti *relativi* tra
configurazioni sullo stesso problema, non come verità assoluta. Il test
oggettivo sui 6 test di riferimento è la metrica primaria; il judge valuta
approccio e completezza sulle soluzioni che i test non distinguono.

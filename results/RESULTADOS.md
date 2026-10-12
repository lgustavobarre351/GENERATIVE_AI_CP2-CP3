# Resultados dos experimentos

Ordenado pelo F1 dos pares (aspecto, sentimento) no teste sintético com frases inéditas.

| Experimento | LoRA r | Camadas | LR | Épocas | Params treináveis | Tempo de treino | Loss val | JSON válido | Acerto exato | F1 pares (teste) | F1 pares (manual) | Acerto exato (manual) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r16_lr2e-4 | 16 | todas | 0.0002 | 1.0 | 17.4M | 32 min | 0.0004 | 100% | 83.0% | **0.926** | 0.913 | 82.5% |
| r64_lr2e-4 | 64 | todas | 0.0002 | 1.0 | 69.7M | 32 min | 0.0005 | 100% | 83.2% | **0.923** | 0.878 | 77.5% |
| r16_lr2e-4_ep2 | 16 | todas | 0.0002 | 2.0 | 17.4M | 64 min | 0.0002 | 100% | 83.2% | **0.920** | 0.905 | 82.5% |
| r16_lr2e-4_attn | 16 | atenção | 0.0002 | 1.0 | 6.4M | 25 min | 0.0055 | 100% | 82.5% | **0.919** | 0.913 | 85.0% |
| r4_lr2e-4 | 4 | todas | 0.0002 | 1.0 | 4.4M | 29 min | 0.0062 | 100% | 80.5% | **0.911** | 0.870 | 75.0% |
| r16_lr1e-3 | 16 | todas | 0.001 | 1.0 | 17.4M | 32 min | 0.0005 | 100% | 76.8% | **0.889** | 0.835 | 67.5% |
| r16_lr2e-5 | 16 | todas | 2e-05 | 1.0 | 17.4M | 30 min | 0.0619 | 100% | 61.8% | **0.831** | 0.837 | 65.0% |
| baseline_zero_shot | — | — | — | — | — | — | — | 100% | 25.5% | **0.514** | 0.611 | 27.5% |

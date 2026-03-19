# Ablation Study — Siamese Bi-LSTM

| Configuration | Accuracy | F1 | AUC |
|---|---|---|---|
| Baseline Bi-LSTM + FastText embeddings | 0.5108 | 0.6755 | 0.5515 |
| + Four-component similarity `[e1, e2, \|e1-e2\|, e1*e2]` | 0.7032 | 0.7509 | 0.8015 |
| + Additive attention mechanism | 0.7440 | 0.7788 | 0.8482 |
| + Deeper classification head | 0.7692 | 0.7916 | 0.8570 |


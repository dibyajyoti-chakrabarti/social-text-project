# Detection results on gold test (n=300)

Point estimates with 95% bootstrap CIs (2,000 resamples). `kappa` = Cohen's κ against the human labels. `s_per_item`: for LLMs, model time per call divided by the 20 items in the call; for local models, inference time on the laptop (RTX 5050 GPU for SetFit). `train_s`: training time in seconds.

| system                    | train           |   macro_f1 | macro_f1_ci   |   accuracy |   kappa |   fr_precision |   fr_recall |   fr_f1 | fr_f1_ci     |   s_per_item |   tokens_per_item |   train_s |
|:--------------------------|:----------------|-----------:|:--------------|-----------:|--------:|---------------:|------------:|--------:|:-------------|-------------:|------------------:|----------:|
| Rules                     | none            |      0.476 | [0.41, 0.53]  |      0.513 |   0.394 |          0.263 |       0.417 |   0.323 | [0.17, 0.47] |        0     |           nan     |   nan     |
| LLM zero-shot (prompt v1) | none            |      0.66  | [0.60, 0.71]  |      0.663 |   0.588 |          0.319 |       0.917 |   0.473 | [0.34, 0.59] |        0.094 |            84.54  |   nan     |
| LLM zero-shot             | none            |      0.693 | [0.63, 0.74]  |      0.713 |   0.638 |          0.383 |       0.75  |   0.507 | [0.35, 0.64] |        0.064 |            92.823 |   nan     |
| LLM few-shot              | 12 dev examples |      0.688 | [0.63, 0.74]  |      0.713 |   0.638 |          0.455 |       0.833 |   0.588 | [0.43, 0.72] |        0.1   |           143.353 |   nan     |
| LLM-2 zero-shot           | none            |      0.677 | [0.61, 0.73]  |      0.72  |   0.636 |          0.536 |       0.625 |   0.577 | [0.40, 0.73] |        0.105 |           157.157 |   nan     |
| TF-IDF+LR (gold-dev)      | 100 gold-dev    |      0.272 | [0.22, 0.32]  |      0.433 |   0.215 |          0.25  |       0.083 |   0.125 | [0.00, 0.29] |        0     |           nan     |     1.893 |
| TF-IDF+SVM (gold-dev)     | 100 gold-dev    |      0.234 | [0.19, 0.28]  |      0.437 |   0.172 |          0     |       0     |   0     | [0.00, 0.00] |        0     |           nan     |     0.038 |
| TF-IDF+NB (gold-dev)      | 100 gold-dev    |      0.301 | [0.25, 0.35]  |      0.41  |   0.225 |          0.226 |       0.292 |   0.255 | [0.09, 0.41] |        0     |           nan     |     0.024 |
| SetFit (gold-dev)         | 100 gold-dev    |      0.601 | [0.54, 0.66]  |      0.667 |   0.569 |          0.4   |       0.333 |   0.364 | [0.17, 0.54] |        0.004 |           nan     |   111.587 |
| TF-IDF+LR (silver)        | 11995 silver    |      0.593 | [0.53, 0.65]  |      0.623 |   0.53  |          0.32  |       0.667 |   0.432 | [0.28, 0.57] |        0     |           nan     |    23.231 |
| TF-IDF+SVM (silver)       | 11995 silver    |      0.6   | [0.54, 0.65]  |      0.627 |   0.532 |          0.441 |       0.625 |   0.517 | [0.35, 0.67] |        0     |           nan     |     2.521 |
| TF-IDF+NB (silver)        | 11995 silver    |      0.544 | [0.48, 0.60]  |      0.617 |   0.505 |          0.522 |       0.5   |   0.511 | [0.32, 0.67] |        0     |           nan     |     1.909 |
| SetFit (silver)           | 11995 silver    |      0.692 | [0.64, 0.74]  |      0.71  |   0.636 |          0.444 |       0.833 |   0.58  | [0.43, 0.71] |        0.004 |           nan     |   462.566 |

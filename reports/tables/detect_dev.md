# Detection: development results on gold dev (n=100)

Few-shot rows exclude the 12 dev items used as examples.

| method           |   n |   s_per_item |   n_scored |   macro_f1 |   accuracy |   fr_precision |   fr_recall |   fr_f1 |   kappa | model                 |   calls |   failed_calls |   tokens_per_item |   train_items |   train_s |
|:-----------------|----:|-------------:|-----------:|-----------:|-----------:|---------------:|------------:|--------:|--------:|:----------------------|--------:|---------------:|------------------:|--------------:|----------:|
| rules            | 100 |        0     |        100 |      0.628 |      0.66  |          0.467 |       0.636 |   0.538 |   0.576 | nan                   |     nan |            nan |            nan    |           nan |   nan     |
| llm_v1_zero_b20  | 100 |      nan     |        100 |      0.671 |      0.65  |          0.375 |       0.818 |   0.514 |   0.575 | gemini-3.5-flash-lite |       5 |              0 |             86.63 |           nan |   nan     |
| llm_v1_few_b20   | 100 |      nan     |         88 |      0.648 |      0.625 |          0.348 |       0.889 |   0.5   |   0.541 | gemini-3.5-flash-lite |       5 |              0 |            106.11 |           nan |   nan     |
| llm_v2_zero_b20  | 100 |      nan     |        100 |      0.718 |      0.73  |          0.5   |       0.727 |   0.593 |   0.664 | gemini-3.5-flash-lite |       5 |              0 |             93.4  |           nan |   nan     |
| llm_v2_few_b20   | 100 |      nan     |         88 |      0.715 |      0.727 |          0.462 |       0.667 |   0.545 |   0.657 | gemini-3.5-flash-lite |       5 |              0 |            114.34 |           nan |   nan     |
| llm_v2_zero_b1   | 100 |      nan     |        100 |      0.743 |      0.75  |          0.562 |       0.818 |   0.667 |   0.69  | gemini-3.5-flash-lite |     100 |              0 |            687.47 |           nan |   nan     |
| llm2_v2_zero_b20 | 100 |      nan     |        100 |      0.759 |      0.78  |          0.667 |       0.727 |   0.696 |   0.72  | openai/gpt-oss-20b    |       5 |              0 |            159.33 |           nan |   nan     |
| llm2_v2_few_b20  | 100 |      nan     |         70 |      0.738 |      0.743 |          0.444 |       0.667 |   0.533 |   0.67  | openai/gpt-oss-20b    |       5 |              1 |            148.27 |           nan |   nan     |
| tfidf_silver     | 100 |        0.001 |        100 |      0.628 |      0.65  |          0.438 |       0.636 |   0.519 |   0.564 | nan                   |     nan |            nan |            nan    |         11995 |    21.369 |
| setfit_silver    | 100 |        0.007 |        100 |      0.719 |      0.75  |          0.538 |       0.636 |   0.583 |   0.688 | nan                   |     nan |            nan |            nan    |         11995 |   469.234 |
| svm_silver_C0.1  | 100 |        0     |        100 |      0.662 |      0.69  |          0.467 |       0.636 |   0.538 |   0.611 | nan                   |     nan |            nan |            nan    |         11995 |     2.908 |
| svm_silver_C0.5  | 100 |        0     |        100 |      0.639 |      0.67  |          0.438 |       0.636 |   0.519 |   0.586 | nan                   |     nan |            nan |            nan    |         11995 |     3.21  |
| svm_silver_C1.0  | 100 |        0     |        100 |      0.644 |      0.67  |          0.467 |       0.636 |   0.538 |   0.584 | nan                   |     nan |            nan |            nan    |         11995 |     3.371 |
| nb_silver_a0.1   | 100 |        0     |        100 |      0.601 |      0.69  |          0.429 |       0.273 |   0.333 |   0.599 | nan                   |     nan |            nan |            nan    |         11995 |     1.952 |
| nb_silver_a0.3   | 100 |        0     |        100 |      0.553 |      0.65  |          0.6   |       0.273 |   0.375 |   0.541 | nan                   |     nan |            nan |            nan    |         11995 |     1.981 |
| nb_silver_a1.0   | 100 |        0     |        100 |      0.546 |      0.65  |          0.5   |       0.182 |   0.267 |   0.534 | nan                   |     nan |            nan |            nan    |         11995 |     1.987 |
| svm_silver_C0.03 | 100 |        0     |        100 |      0.664 |      0.68  |          0.5   |       0.636 |   0.56  |   0.602 | nan                   |     nan |            nan |            nan    |         11995 |     2.431 |
| nb_silver_a0.03  | 100 |        0     |        100 |      0.632 |      0.71  |          0.5   |       0.364 |   0.421 |   0.627 | nan                   |     nan |            nan |            nan    |         11995 |     1.928 |
| svm_silver_C0.01 | 100 |        0     |        100 |      0.649 |      0.66  |          0.636 |       0.636 |   0.636 |   0.58  | nan                   |     nan |            nan |            nan    |         11995 |     2.546 |
| nb_silver_a0.01  | 100 |        0     |        100 |      0.61  |      0.7   |          0.571 |       0.364 |   0.444 |   0.613 | nan                   |     nan |            nan |            nan    |         11995 |     1.915 |

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
| tfidf_silver     | 100 |        0.001 |        100 |      0.656 |      0.68  |          0.438 |       0.636 |   0.519 |   0.597 | nan                   |     nan |            nan |            nan    |          8880 |    20.214 |
| setfit_silver    | 100 |        0.007 |        100 |      0.738 |      0.76  |          0.467 |       0.636 |   0.538 |   0.701 | nan                   |     nan |            nan |            nan    |          8880 |   461.778 |

# Extraction evaluation

**Gold**: the 35 gold feature requests, against the annotator's feature phrases (34 items have one). Phrases match when the cosine similarity of their `all-MiniLM-L6-v2` embeddings is ≥ 0.6 (one-to-one); `f1@x` shows other thresholds. `token_f1`: best lemma overlap per gold phrase. `item_hit`: items with at least one matched phrase. `empty`: items with no extracted phrase.

**Corpus sample**: 100 random corpus items the deployed classifier labels feature_request (gold excluded), graded from the text alone, so classifier false positives count against a method unless it extracts nothing.

`judge_*`: share of items graded correct / partial / wrong by openai/gpt-oss-20b (blind to the method; for gold it also sees the annotator's phrases). Extractor: gemini-3.5-flash-lite.

| setting       | method   |   items |   precision |   recall |      f1 | f1_ci        |   f1@0.5 |   f1@0.7 |   token_f1 |   item_hit |   empty |   phrases_per_item |   judge_correct |   judge_partial |   judge_wrong |
|:--------------|:---------|--------:|------------:|---------:|--------:|:-------------|---------:|---------:|-----------:|-----------:|--------:|-------------------:|----------------:|----------------:|--------------:|
| gold all      | rules    |      34 |       0.458 |    0.314 |   0.373 | [0.20, 0.55] |    0.407 |    0.339 |      0.248 |      0.324 |   0.441 |              0.706 |           0.343 |           0.114 |         0.543 |
| gold all      | llm      |      34 |       0.391 |    0.514 |   0.444 | [0.30, 0.59] |    0.543 |    0.37  |      0.363 |      0.529 |   0.088 |              1.353 |           0.629 |           0.171 |         0.2   |
| gold dev      | rules    |      11 |       0.333 |    0.182 |   0.235 | [0.00, 0.53] |    0.235 |    0.235 |      0.172 |      0.182 |   0.545 |              0.545 |           0.273 |           0.091 |         0.636 |
| gold dev      | llm      |      11 |       0.231 |    0.273 |   0.25  | [0.00, 0.50] |    0.333 |    0.25  |      0.2   |      0.273 |   0.091 |              1.182 |           0.727 |           0.182 |         0.091 |
| gold test     | rules    |      23 |       0.5   |    0.375 |   0.429 | [0.23, 0.63] |    0.476 |    0.381 |      0.283 |      0.391 |   0.391 |              0.783 |           0.375 |           0.125 |         0.5   |
| gold test     | llm      |      23 |       0.455 |    0.625 |   0.526 | [0.36, 0.69] |    0.632 |    0.421 |      0.438 |      0.652 |   0.087 |              1.435 |           0.583 |           0.167 |         0.25  |
| corpus sample | rules    |     100 |     nan     |  nan     | nan     | nan          |  nan     |  nan     |    nan     |    nan     |   0.51  |              0.68  |           0.24  |           0.16  |         0.6   |
| corpus sample | llm      |     100 |     nan     |  nan     | nan     | nan          |  nan     |  nan     |    nan     |    nan     |   0.16  |              1.12  |           0.83  |           0.06  |         0.11  |

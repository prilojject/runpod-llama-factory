# Bundled example datasets (MIT)

`tiny_sft.json` (16 alpaca rows), `tiny_dpo.json` (8 preference rows) and `tiny_kto.json`
(16 rows with `kto_tag`) were written from scratch for this repository and are released under
the repository's MIT License (see `../../LICENSE`). They contain no text from Stanford Alpaca
(CC BY-NC 4.0) or any other third-party corpus.

Use them by name (`"dataset": "tiny_sft"`) or by URL
(`https://raw.githubusercontent.com/prilojject/runpod-llama-factory/main/examples/datasets/tiny_sft.json`).
They are for smoke tests only: 16 rows teach a model nothing useful.

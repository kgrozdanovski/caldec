# Local build records

The data-generation recipe stores per-build protocols, API call journals, cached responses, annotations, rejected candidates and freeze hashes in this directory. Git ignores those records; they are not part of the published dataset or Hub upload.

`scripts/generate.py` writes one subdirectory per named build. After `freeze` verifies the labels and hashes the accepted cases, `fold` appends its rows to the selected JSONL split of the build's working dataset directory (`data_dir`, for example `data-next/`). It never writes to the v1 split files in `data/`. The split rows retain the six-field public schema. A build name prefixes its case ids, so `scripts/data.py` can select those rows with `load_split(..., build="<build>")` when needed.

Do not upload this directory. The Hub dataset repository holds only the three split files, `sites.json`, `prompts.json`, and the dataset card.

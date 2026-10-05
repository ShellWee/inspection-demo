# COBBIE IFC evaluation and the frozen Text-GNN v5 main system

This repository retains the upstream COBBIE implementation and provides the
current, query-free integration used by the Text-GNN v5 paper system. The main
system is:

```text
clean ToG traversal
+ frozen Text-GNN v5 retrieval plugin
+ typed hierarchy
+ evidence closure v3
```

The three ToG-family arms share one traversal engine and differ only in the
declared switches below.

| Arm | GNN | Typed hierarchy | Closure |
|---|---:|---:|---|
| ToG-BIM | off | off | legacy ToG finalization |
| ToG-GNN | on | off | legacy ToG finalization |
| ToG-Hierarchy | on | on | `closure-adjudication-v3` |

All arms execute the original ToG sequence: model planning, LLM relation
selection, graph traversal, LLM entity pruning, sufficiency checking, and at
least one LLM answering call. ToG-BIM never imports or calls the GNN plugin.
The public current adapter accepts raw questions and IFC paths, not evaluator
category, gold answers, benchmark labels, or question-ID routing.

## Frozen identities

- COBBIE package version: `0.2.0`
- ToG profile: `paper-v14-clean-v1`
- retrieval profile: `query-conditioned-plan-v5.0`
- plugin entry point: `text-gnn-v5.0`
- Text-GNN v5 checkpoint SHA-256:
  `3349b9cb992783806fffa59d57f84e8e93a7d2d1faea08738209e438b246d743`
- runtime manifest SHA-256:
  `81fdd5bf5de079e7067f4cc3fb4775348466fa994333b6ce462aafa392948309`
- inspection graph SHA-256:
  `3bc15485690c3911fa6dccdb7ad7e0a9672f4cf84e82d3fc58025af159107f19`
- runtime asset SHA-256:
  `eb5c4e06c061313ffde0d511c11bf0caa855214afdce6773d1035fcf10131cdc`
- Ecore IFC SHA-256:
  `84918ca0cd1f9301411ec9350f175eff1519cfc3e8a9a0c4ddbebf2b6787db79`

Exact repository refs and artifact hashes are recorded in
`releases/paper-v5/release-lock.json`.

## Fresh-clone bootstrap

Python 3.12 and `uv` 0.11.19 are required. The private GitHub repositories and
release asset require either an authenticated GitHub CLI session or a
repository-scoped `GITHUB_TOKEN`/`GH_TOKEN`.

```bash
git clone --branch cobbie-text-gnn-v5.0.2-2026-09-02 \
  https://github.com/ShellWee/cobbie-ecore.git bootstrap-source

python bootstrap-source/scripts/bootstrap_paper_v5.py \
  --workspace ./paper-v5-workspace
```

The bootstrap command performs all of the following without editing source:

1. clones the three immutable release tags;
2. downloads and verifies the private Text-GNN runtime asset;
3. runs `uv sync --frozen --only-group paper-v5`;
4. installs the two sibling packages as no-dependency editable checkouts;
5. runs the real plugin, packaged graph, and ToG-BIM offline smoke;
6. regenerates BAML in a temporary directory and checks exact parity; and
7. verifies all three Git checkouts remain clean.

An already downloaded asset can be supplied with `--asset`; its frozen hash is
still checked.

## Offline verification

From the bootstrapped COBBIE checkout:

```bash
uv run --no-sync python scripts/run_tog_gnn_smoke.py \
  --runtime ../runtime/text-gnn-v5.0-runtime \
  --output ../verification/offline-smoke.json

uv run --no-sync python scripts/verify_baml_generated.py
```

The smoke is credential-free and makes no network calls. It verifies the real
plugin entry point, standalone/plugin top-50 equality, the top-20 operational
and top-50 evaluation boundary, finite scores, the packaged graph hash,
ToG-BIM plugin isolation, and a mandatory answering call through a deterministic
fixture LLM.

## Live Ecore evaluation

The runtime asset is intentionally query-free. It contains no questions, gold
answers, evaluator metadata, IFC file, or credentials. A lawful Ecore IFC and
the protected evaluation database must therefore be supplied outside Git. The
database can remain anywhere; no source file needs to be changed.

```bash
export ECORE_IFC_PATH=/absolute/path/to/ecore.ifc
export COBBIE_DB_PATH=/absolute/path/to/ecore-evaluation.db
export OPENAI_API_KEY=...
```

Use the frozen 59-ID manifest at `data/ecore_question_ids.json`. For ToG-BIM:

```bash
uv run --no-sync python scripts/run_evaluation.py \
  --system tog-bim \
  --question-ids data/ecore_question_ids.json \
  --client OpenAI_GPT_4_1 \
  --judge-client OpenAI_GPT_4_1 \
  --output-results outputs/paper-v5/tog-bim.json
```

For ToG-GNN:

```bash
uv run --no-sync python scripts/run_evaluation.py \
  --system tog-bim-gnn \
  --question-ids data/ecore_question_ids.json \
  --client OpenAI_GPT_4_1 \
  --judge-client OpenAI_GPT_4_1 \
  --tog-gnn-artifact-dir ../runtime/text-gnn-v5.0-runtime \
  --tog-retriever-plugin text-gnn-v5.0 \
  --tog-retriever-plugin-manifest-sha256 81fdd5bf5de079e7067f4cc3fb4775348466fa994333b6ce462aafa392948309 \
  --tog-retriever-plugin-runtime-graph-sha256 3bc15485690c3911fa6dccdb7ad7e0a9672f4cf84e82d3fc58025af159107f19 \
  --tog-retriever-plugin-device cpu \
  --output-results outputs/paper-v5/tog-gnn.json
```

For the paper main system, append:

```bash
  --tog-hierarchy-reasoning \
  --tog-retrieval-flow-profile closure-adjudication-v3
```

Set `--tog-retriever-plugin-device cuda` only in a CUDA-capable environment.
`ECORE_IFC_PATH` overrides workstation-specific model paths stored in the
external database. API keys are read only from process environment and must
never be committed.

## Current implementation boundaries

- Text-GNN uses top-20 anchors operationally and preserves top-50 only for
  evaluation and bounded typed-path promotion.
- Hierarchy follows real typed edges. Support nodes are not promoted to targets
  without an authoritative complete typed path and binding eligibility.
- Closure v3 certifies target, scope, relation, direction, cardinality, and
  action ordering before deterministic finalization.
- Predicted edges are not injected into the runtime graph.
- The runtime artifact and source are query-free. Ecore results remain
  in-domain/post-selection evaluation because system design was repeatedly
  inspected against that benchmark; this is distinct from training leakage.

## Production-v14 historical reproduction

The immutable historical tag `paper-v14-repro-2026-08-14` remains unchanged.
Its source and four release archives are verified by:

```bash
python scripts/bootstrap_paper_v14.py --workspace ../paper-v14-workspace
python scripts/verify_paper_v14.py --offline \
  --artifacts-dir ../paper-v14-workspace/artifacts/paper-v14
```

The production-v14 checkpoint SHA-256 is
`49faf5f5596e972c033b6fa831d09d735a7d930c9ccd752053f6cb471ded806b`.
The historical tag is the authoritative source for retired v3/v4 integration
code; those implementations are not exposed by the current package.

## Development checks

```bash
uv run python -m pytest -q -p no:cacheprovider tests
uv run ruff check src scripts tests
uv run python scripts/verify_baml_generated.py
git diff --check
```

CI runs the current release surface on Ubuntu and Windows with Python 3.12.
CUDA is not required in GitHub CI; local release validation separately checks
CPU/CUDA score tolerance and top-50 parity.

## Upstream attribution and data

This fork retains the upstream COBBIE Git history, attribution, and CC BY 4.0
license. The public IFC-Bench dataset is maintained separately; this repository
does not redistribute the Ecore IFC or protected evaluation rows. See
`docs/UPSTREAM.md`, `docs/VERSIONING.md`, `docs/REPRODUCIBILITY.md`, and
`docs/ifc-bench.md` for provenance and dataset setup.

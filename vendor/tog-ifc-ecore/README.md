# ToG-IFC: paper-v14 clean traversal with Text-GNN v5

This repository provides the provider-independent IFC Think-on-Graph runtime
used by the Text-GNN v5 paper system. The canonical `ToGSystem` preserves the original ToG loop—LLM
relation selection, graph expansion, LLM entity pruning, sufficiency checks,
and a mandatory grounded answer call—while exposing a generic retriever plugin
interface and the current typed-hierarchy closure.

The upstream projects are [DataArcTech/ToG](https://github.com/DataArcTech/ToG)
and [GasolSun36/ToG](https://github.com/GasolSun36/ToG). Their history and
attribution are retained. The historical Production-v14 reproduction remains at
tag `paper-v14-repro-2026-08-14`.

## Current family

All variants consume only the raw question and IFC model path:

```python
response = system.ask(question, model_path=ifc_path)
```

- **ToG-BIM:** original graph traversal with leakage-safe inputs; no GNN plugin
  and no hierarchy closure.
- **ToG-GNN:** the same traversal plus frozen Text-GNN v5 operational evidence.
- **ToG-Hierarchy:** the same v5 ranking plus authoritative typed paths and
  closure-adjudication-v3.

Evaluator category, gold, question ID, benchmark labels, and answer hints are
not accepted by the public system call. ToG-BIM cannot load the GNN plugin.
The retired standalone `clean_engine` is available from Git history but is not
an alternate package entry point; it lacked the closure-v3 execution path.

## Install and test

Python 3.12 is supported.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m tog --help
```

Text-GNN is loaded through entry-point group `tog_ifc.retrievers`; the current
plugin ID is `text-gnn-v5.0`. A hierarchy configuration without an active GNN
plugin fails closed.

## Release contract

The v5 integration release is tagged
`tog-ifc-text-gnn-v5.0.0-2026-09-02`. The COBBIE paper-v5 bootstrap pins this
tag and runs the synthetic credential-free smoke. Live evaluation requires an
external IFC path and provider credential; neither is stored here.

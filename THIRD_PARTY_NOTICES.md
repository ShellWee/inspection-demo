# Source and asset provenance

This private repository is a runnable application snapshot, not a blanket
relicensing of the research projects or Ecore dataset.

- `vendor/cobbie-ecore` retains its upstream README and CC BY 4.0 LICENSE.
  Preserve its attribution and notices when redistributing permitted portions.
- `vendor/tog-ifc-ecore` retains its README, project metadata and source. Upstream
  references are recorded there and in `SOURCE_MANIFEST.json`. This repository
  does not grant rights missing from those original materials.
- `vendor/text-gnn-plugin` registers the runtime plugin. Model code and weights
  arrive separately in the hash-verified private asset bundle.
- `SOURCE_MANIFEST.json` is historical research snapshot provenance, not a claim
  that its hashes describe subsequent application fixes. `RELEASE_MANIFEST.json`
  inventories the current reviewed source, notebook and prebuilt UI bytes.
- Ecore IFC, graph/index, model weights and floor-plan artifacts are **not in
  Git**. `assets.lock.json` identifies a pinned private dataset revision;
  `backend/inspection_demo/expected_assets.json` verifies the exact payloads.
  Repository access does not confer dataset access or redistribution rights.
- Python/JavaScript dependencies and fonts keep their respective upstream
  licenses. The frontend uses IBM Plex and Saira Semi Condensed font packages;
  bundled notices are retained in `frontend/THIRD_PARTY_LICENSES.txt`.

Confirm all source and asset redistribution permissions before making this
repository or its derived building information public. Do not commit API keys,
HF tokens, user runs, notebook outputs or local credentials.

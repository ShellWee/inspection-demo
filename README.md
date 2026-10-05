
# Ecore Inspection Target Planning

## Git + Notebook quick start

Official source repository: [ShellWee/inspection-demo](https://github.com/ShellWee/inspection-demo). This source release includes the newer
answer/planning distinction and what-if validation fixes.

## Usage
```bash
git clone https://github.com/ShellWee/inspection-demo.git
cd inspection-demo
```

Open **[notebooks/Launch_Demo.ipynb](notebooks/Launch_Demo.ipynb)** in VS Code or
Jupyter and execute its cells in order. The notebook kernel needs an approved
Python 3.10+ interpreter; setup creates a separate locked Python 3.12 application
environment. The built React UI is included: **Node.js is not required to run**.
If you need a notebook editor, install JupyterLab in your approved editor/kernel
environment, not in the app environment that `uv sync` manages.

The notebook installs dependencies, verifies assets, starts the local server,
checks status, optionally submits a query and stops the server. Setup and start
do not call an OpenAI model.

- Assets: approximately **943 MB**, downloaded separately from the private HF
  dataset revision pinned in `assets.lock.json`. Obtain read access and use
  `hf auth login`, or point `ASSETS_DIR` to an existing complete asset bundle.
- Local Jupyter/VS Code: open the printed localhost URL for the full three-stage UI.
- Colab: use the notebook's loopback HTTP query cells. This is **not** public web
  hosting, and no tunnel or background keep-alive workaround is created.
- CPU is the default; CUDA is optional on compatible Linux hosts. Linux currently
  uses the locked CUDA-enabled Torch distribution even in CPU mode, so dependency
  downloads are substantially larger than the asset bundle. Disk requirements
  also include the extracted model and copied graph index.
- The app still uses the existing single-user ownership model. Do not expose
  its port publicly. This release does not implement the proposed remote ZeroGPU
  service or change cloud billing.

See [step-by-step notebook instructions](docs/NOTEBOOK_QUICKSTART.md) and
[source/asset notices](THIRD_PARTY_NOTICES.md).

### Updating an existing checkout

```bash
git pull --ff-only
```

Stop the existing demo first, preserve your uncommitted changes, then rerun setup
and asset verification from the notebook. A checkout is identified by its Git
commit; `RELEASE_MANIFEST.json` checks the shipped bytes. Do not copy a Colab/Linux
virtual environment onto Windows.

### Maintainer checks (Python 3.12+, no model/API calls)

```bash
python -m unittest discover -s scripts/tests -v
python scripts/verify_release.py
python scripts/repository_manifest.py --check
```

After editing source or the UI, rebuild the frontend with the locked package
manager environment, review the diff, clear notebook outputs, then regenerate
the inventory with `python scripts/repository_manifest.py --write`. The historical
`SOURCE_MANIFEST.json` is preserved as research provenance. Include the current
inventory with each release; do not commit downloaded assets or state directories.


## Space configuration

- Hardware: `l4x1`
- Visibility: private
- Read-only volume: `hf://datasets/<namespace>/inspection-demo-ecore-assets-v1:/mnt/default-assets:ro`
- One Uvicorn worker on port 7860

The entrypoint also supports `HF_ASSET_DATASET` plus an `HF_TOKEN` secret as a
fail-closed fallback when native Dataset mounting is unavailable. The normal
deployment uses the read-only volume and does not expose a Hub token to the
container.

After `hf auth login`, deploy both private repositories and attach the L4 with:

```powershell
.\scripts\deploy_hf.ps1 -Namespace <hugging-face-username>
```

An `l4x1` Space requires Hugging Face prepaid credits. The deployment script
reserves the Space before uploading the large asset bundle and stops immediately
on any non-zero CLI exit code.

No OpenAI API key is persisted. Run state, events, trajectories, and replays use ephemeral storage and disappear when the Space restarts.

## Local verification

The UI now uses a simplified three-stage inspection workspace. See
[refinement notes](docs/REFINEMENT.md) for changes, defect fixes and cost-limited
test instructions. Internal pipeline identifiers remain in engineering artifacts,
not the main user interface.

```bash
uv sync --frozen --extra dev
uv run pytest
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend test
pnpm --dir frontend build
docker build -t inspection-demo-hf .
```

import sys

import pytest
from inspection_demo.grounding import SubprocessResearchGroundingAdapter


@pytest.mark.asyncio
async def test_research_adapter_passes_key_only_over_stdin_and_validates_result(tmp_path) -> None:
    bridge = tmp_path / "bridge.py"
    bridge.write_text(
        "import json,sys\n"
        "p=json.load(sys.stdin)\n"
        "assert p['api_key']=='memory-only'\n"
        "print(json.dumps({'status':'abstained','answer':'','closure_status':'abstain',"
        "'closure_stop_reason':'fixture','tasks':[],'reasoning':[],'subgraph':None,"
        "'errors':[],'executable':False}))\n",
        encoding="utf-8",
    )
    adapter = SubprocessResearchGroundingAdapter(
        python_executable=sys.executable,
        bridge_path=bridge,
        base_payload={"fixed": True},
        timeout_seconds=5,
    )

    result = await adapter.ground(
        query="Inspect any available terminal.",
        model_id="gpt-4.1",
        api_key="memory-only",
    )

    assert result.status == "abstained"
    assert result.closure_stop_reason == "fixture"


@pytest.mark.asyncio
async def test_worker_error_redacts_api_key(tmp_path) -> None:
    bridge = tmp_path / "bridge.py"
    bridge.write_text(
        "import json,sys\np=json.load(sys.stdin)\n"
        "sys.stderr.write('provider rejected '+p['api_key'])\nraise SystemExit(2)\n",
        encoding="utf-8",
    )
    adapter = SubprocessResearchGroundingAdapter(
        python_executable=sys.executable,
        bridge_path=bridge,
        base_payload={},
        timeout_seconds=5,
    )
    with pytest.raises(RuntimeError) as caught:
        await adapter.ground(
            query="Inspect target.", model_id="gpt-4.1", api_key="memory-only-secret"
        )
    assert "memory-only-secret" not in str(caught.value)

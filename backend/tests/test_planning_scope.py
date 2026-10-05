import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor/tog-ifc-ecore/src"))
from tog import retrieval_closure
from tog.models import QueryPlan
from tog.planning import merge_query_plan


def test_semantic_plan_cannot_reintroduce_source_room_as_target_scope():
    fallback = QueryPlan(operator="path", room=None, room_names=[])
    proposed = QueryPlan(operator="path", room="A12", room_names=["A12"])
    merged = merge_query_plan(fallback, proposed)
    assert merged.room is None
    assert merged.room_names == []


def test_contract_view_masks_input_identifiers_without_losing_reference_identity():
    question = (
        "If pump `1234567890123456789012` stops, compare it with "
        "ifc_2234567890123456789012 and 1234567890123456789012."
    )
    view = retrieval_closure.contract_question_view(question)
    assert "1234567890123456789012" not in view
    assert "2234567890123456789012" not in view
    assert view.count("[entity reference 1]") == 2
    assert view.count("[entity reference 2]") == 1
    assert "If pump" in view


def test_reviewer_rejection_cannot_be_overridden_by_retaining_initial_groups():
    from types import SimpleNamespace

    rc = retrieval_closure
    packet = SimpleNamespace(verify=lambda: True, operator_result={}, packet_sha256="packet")
    contract = rc.RetrievalIntentContractV3(
        compact_plan_sha256="plan",
        bindings=[
            {
                "binding_index": 0,
                "action": "Inspect",
                "selection_mode": "all_matching",
                "target_kind": "object",
            }
        ],
    ).seal()
    ledger = rc.CandidateGroupLedgerV3(
        packet_sha256="packet",
        compact_plan={},
        contract={},
        source_diagnostics={},
        candidates=[
            {
                "candidate_alias": "B00C001",
                "node_id": "ifc_fixture",
                "eligibility": "reviewable",
                "union_rank": 1,
            }
        ],
        groups=[
            {
                "group_alias": "B00G001",
                "binding_index": 0,
                "eligibility": "reviewable",
                "candidate_aliases": ["B00C001"],
            }
        ],
    ).seal()
    initial = rc.GroupSelectionV3(
        ledger_sha256=ledger.ledger_sha256,
        closure_supported=True,
        binding_selections=[
            {"binding_index": 0, "selected_group_ids": ["B00G001"], "uncertain_group_ids": []}
        ],
    ).seal()
    review = rc.GroupSelectionV3(
        ledger_sha256=ledger.ledger_sha256,
        closure_supported=True,
        provider_summary="No dependency path is supported.",
        binding_selections=[
            {
                "binding_index": 0,
                "selected_group_ids": [],
                "uncertain_group_ids": [],
                "reason": "Missing dependency evidence",
            }
        ],
    ).seal()
    selection = rc.reconcile_group_selections_v3(packet, contract, ledger, initial, review)
    certificate = rc.adjudicate_group_selection_v3(packet, contract, ledger, selection)
    assert certificate.closure_status == "abstain"
    assert certificate.stop_reason == "provider_selection_disagreement"
    assert "No dependency path" in selection.provider_summary

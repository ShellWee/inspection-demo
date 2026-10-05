from types import SimpleNamespace

from inspection_demo import research_bridge


def test_intent_classifier_preserves_question_and_accounts_for_tokens():
    import json

    import httpx
    from openai import OpenAI

    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"kind":"answer",'
                                '"reason":"Requests an explanation of possible impact."}'
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
            },
        )

    query = "If a valve closes, which rooms might lose supply?"
    with OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    ) as client:
        intent, usage = research_bridge._classify_query(query, "gpt-4.1", client)
    assert intent.kind == "answer"
    assert seen[0]["messages"][-1]["content"] == query
    assert usage == {
        "llm_calls": 1,
        "input_tokens": 120,
        "output_tokens": 20,
        "cached_input_tokens": 0,
    }


def test_refused_classification_never_defaults_to_execution():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                parse=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))], usage=None
                )
            )
        )
    )
    intent, usage = research_bridge._classify_query("A request", "gpt-4.1", client)
    assert intent.kind == "unknown"
    assert usage["llm_calls"] == 1

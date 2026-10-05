from types import SimpleNamespace

import pytest
from inspection_demo.openai_catalog import OpenAIModelCatalog


class FakeModels:
    async def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-4.1"),
                SimpleNamespace(id="gpt-5"),
                SimpleNamespace(id="gpt-5.6-luna"),
                SimpleNamespace(id="gpt-5.6-terra"),
                SimpleNamespace(id="text-embedding-3-small"),
            ]
        )


class FakeResponses:
    def __init__(self) -> None:
        self.probed: list[str] = []

    async def create(self, **request: object) -> SimpleNamespace:
        model = str(request["model"])
        self.probed.append(model)
        return SimpleNamespace(id="resp_probe")


class FakeOpenAI:
    def __init__(self) -> None:
        self.models = FakeModels()
        self.responses = FakeResponses()


@pytest.mark.asyncio
async def test_catalog_only_returns_the_three_approved_models_without_paid_probes() -> None:
    """Catches unrelated models and paid generation calls leaking into Connect models."""
    client = FakeOpenAI()
    catalog = OpenAIModelCatalog(client_factory=lambda _: client)

    options = await catalog.list("sk-test")
    by_id = {option.id: option for option in options}

    assert list(by_id) == ["gpt-4.1", "gpt-5", "gpt-5.6-luna"]
    assert client.responses.probed == []
    assert by_id["gpt-4.1"].compatibility == "validated"
    assert all(by_id[model].compatibility == "compatible" for model in ("gpt-5", "gpt-5.6-luna"))
    assert by_id["gpt-4.1"].recommended is True


@pytest.mark.asyncio
async def test_catalog_falls_back_to_gpt5_when_gpt41_is_unavailable() -> None:
    client = FakeOpenAI()

    async def list_without_gpt41() -> SimpleNamespace:
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-5"),
                SimpleNamespace(id="gpt-5.6-luna"),
            ]
        )

    client.models.list = list_without_gpt41
    catalog = OpenAIModelCatalog(client_factory=lambda _: client)

    options = await catalog.list("sk-test")

    assert options[0].id == "gpt-5"
    assert options[0].recommended is True

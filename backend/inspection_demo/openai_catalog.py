from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .models import ModelOption


@dataclass(frozen=True)
class ModelProfile:
    label: str
    client_kind: str
    embedding_model: str = "text-embedding-3-small"


MODEL_PROFILES = {
    "gpt-4.1": ModelProfile(label="GPT-4.1", client_kind="baml"),
    "gpt-5": ModelProfile(label="GPT-5", client_kind="responses"),
    "gpt-5.6-luna": ModelProfile(label="GPT-5.6 Luna", client_kind="responses"),
}


class OpenAIModelCatalog:
    def __init__(
        self,
        *,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.client_factory = client_factory or (
            lambda api_key: AsyncOpenAI(api_key=api_key, timeout=20.0, max_retries=0)
        )

    async def list(self, api_key: str) -> list[ModelOption]:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required")
        client = self.client_factory(api_key)
        try:
            page = await client.models.list()
            available_ids = {model.id for model in page.data}
            ids = [model_id for model_id in MODEL_PROFILES if model_id in available_ids]
            options = [self._classify(model_id) for model_id in ids]
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result

        compatible_ids = {option.id for option in options if option.compatibility != "unsupported"}
        recommended_id = next(
            (model_id for model_id in MODEL_PROFILES if model_id in compatible_ids),
            None,
        )
        options = [
            option.model_copy(update={"recommended": option.id == recommended_id})
            for option in options
        ]
        return sorted(
            options,
            key=lambda option: (
                option.compatibility == "unsupported",
                not option.recommended,
                option.id,
            ),
        )

    def _classify(self, model_id: str) -> ModelOption:
        if model_id in MODEL_PROFILES:
            return ModelOption(
                id=model_id,
                label=MODEL_PROFILES[model_id].label,
                compatibility="validated" if model_id == "gpt-4.1" else "compatible",
                reason="Approved for Ecore inspection grounding.",
            )
        return ModelOption(
            id=model_id,
            label=model_id,
            compatibility="unsupported",
            reason="This model is outside the inspection demo allowlist.",
        )

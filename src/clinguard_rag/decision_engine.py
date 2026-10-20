"""Unified System 1 Decision Engine supporting Laya SDK, /v1/systemone HTTP, and Golden Replay."""

from __future__ import annotations
import os
import re
from typing import Any, Callable, Dict, List, Optional
import httpx


def redact_phi_hook(state: str) -> str:
    """Prediction hook that scrubs email and phone patterns before inference."""
    scrubbed = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", state)
    scrubbed = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", scrubbed)
    return scrubbed


class SystemOneEngine:
    """Base protocol for Laya (local open-weight) and Jev (/v1/systemone API) backends."""

    backend_name: str = "base"
    checkpoint_id: str = "unknown"

    def predict_case(self, case_record: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class LayaLocalBackend(SystemOneEngine):
    """
    Wraps the official `laya` Python SDK (Apache-2.0, Convai Innovations).
    Sets USE_TF=0 prior to importing laya to prevent transformers/abseil deadlocks.
    """

    backend_name = "laya-local-sdk"

    def __init__(
        self,
        checkpoint_id: str = "convaiinnovations/laya-typed-decisions",
        pre_hook: Optional[Callable[[str], str]] = redact_phi_hook,
    ) -> None:
        os.environ["USE_TF"] = "0"
        self.checkpoint_id = checkpoint_id
        self.pre_hook = pre_hook
        import laya  # Requires optional `pip install laya`

        self._agent = laya.load(checkpoint_id)

    def predict_raw(self, state: str, questions: List[Dict[str, Any]]) -> Any:
        clean_state = self.pre_hook(state) if self.pre_hook else state
        return self._agent.system_one(state=clean_state, questions=questions)


class SystemOneHTTPBackend(SystemOneEngine):
    """
    HTTP client for TypeSafe AI's Jev (`POST /v1/systemone`) or self-hosted `laya-serve`.
    Switching between Jev Cloud and local `laya-serve` requires only changing `base_url`.
    """

    backend_name = "systemone-http-api"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        checkpoint_id: str = "laya-serve-or-jev",
        timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LAYA_API_KEY")
        self.checkpoint_id = checkpoint_id
        self.timeout_s = timeout_s

    def predict_raw(self, state: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        clean_state = redact_phi_hook(state)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"state": clean_state, "questions": questions}
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/v1/systemone", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()


class DeterministicEvalBackend(SystemOneEngine):
    """
    Golden-fixture replay and heuristic System 1 backend for reproducible offline evaluation
    and fast CI testing without requiring an 843 MB model download.
    """

    backend_name = "deterministic-golden-eval"

    def __init__(self, checkpoint_id: str = "convaiinnovations/laya-typed-decisions-golden") -> None:
        self.checkpoint_id = checkpoint_id

    def predict_case(self, case_record: Dict[str, Any]) -> Dict[str, Any]:
        clean_query = redact_phi_hook(case_record["query"])
        probs = dict(case_record["raw_model_probs"])
        probs["sanitized_query"] = clean_query
        return probs
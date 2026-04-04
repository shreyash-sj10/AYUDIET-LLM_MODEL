import time

from fastapi.testclient import TestClient

import main


class DummyWrapper:
    def __init__(self, profile=None, feedback=None, explanation=None, sleep_sec=0):
        self.profile = profile or {"success": True, "data": {"risk_flags": ["none"], "dosha_estimate": {"vata": 0.33, "pitta": 0.33, "kapha": 0.34}, "confidence": 0.5}}
        self.feedback = feedback or {"success": True, "data": {"feedback_type": "LIKE", "target": "recipe"}}
        self.explanation = explanation or {"success": True, "data": {"explanation": "context based explanation", "sources": ["source"]}}
        self.sleep_sec = sleep_sec

    def extract_health_profile(self, *_args, **_kwargs):
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        return self.profile

    def parse_feedback(self, *_args, **_kwargs):
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        return self.feedback

    def generate_explanation(self, *_args, **_kwargs):
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        return self.explanation


def test_profile_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 60)
    monkeypatch.setattr(main, "_RATE_BUCKETS", {})
    monkeypatch.setattr(main, "_get_strict_wrapper", lambda: DummyWrapper())

    client = TestClient(main.app)
    res = client.post("/ai/profile", json={"text": "I have acidity"})
    assert res.status_code == 401

    res_ok = client.post(
        "/ai/profile",
        json={"text": "I have acidity"},
        headers={"X-API-Key": "secret"},
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["success"] is True


def test_rate_limit_enforced(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(main, "_RATE_BUCKETS", {})
    monkeypatch.setattr(main, "_get_strict_wrapper", lambda: DummyWrapper())

    client = TestClient(main.app)
    first = client.post("/ai/feedback", json={"text": "good"})
    second = client.post("/ai/feedback", json={"text": "good"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_timeout_falls_back(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 60)
    monkeypatch.setattr(main, "_RATE_BUCKETS", {})
    monkeypatch.setattr(main, "LLM_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main, "_get_strict_wrapper", lambda: DummyWrapper(sleep_sec=0.2))

    client = TestClient(main.app)
    res = client.post("/ai/profile", json={"text": "I have acidity"})
    body = res.json()

    assert res.status_code == 200
    assert body["success"] is True
    assert body["data"]["risk_flags"] == ["none"]


def test_explain_context_mismatch_fallback(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 60)
    monkeypatch.setattr(main, "_RATE_BUCKETS", {})
    wrapper = DummyWrapper(
        explanation={
            "success": True,
            "data": {
                "explanation": "quantum liver protocol always works",
                "sources": ["x"],
            },
        }
    )
    monkeypatch.setattr(main, "_get_strict_wrapper", lambda: wrapper)

    client = TestClient(main.app)
    res = client.post(
        "/ai/explain",
        json={"context": "Pitta relates to heat", "reasoning": "heat symptoms"},
    )

    assert res.status_code == 200
    assert "Insufficient context" in res.json()["data"]["explanation"] or res.json()["data"]["explanation"] == "Pitta relates to heat"

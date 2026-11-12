"""Endpoints and session checks (SPEC CA-07, CA-08)."""

from __future__ import annotations

import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import processing
from app.core.config import get_settings
from app.dependencies.auth import get_jwks_client

SUPABASE_URL = "https://example.supabase.co"
ISSUER = f"{SUPABASE_URL}/auth/v1"
USER_ID = "11111111-1111-1111-1111-111111111111"
DOCUMENT_ID = "22222222-2222-2222-2222-222222222222"

PROJECT_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())


class FakeJwksClient:
    """Always answers with the project's public key, like the real JWKS endpoint."""

    def get_signing_key_from_jwt(self, token):
        return jwt.PyJWK.from_dict(
            {**jwt.algorithms.ECAlgorithm.to_jwk(PROJECT_KEY.public_key(), as_dict=True), "alg": "ES256"}
        )


def token(key=PROJECT_KEY, **overrides) -> str:
    claims = {
        "sub": USER_ID,
        "aud": "authenticated",
        "role": "authenticated",
        "iss": ISSUER,
        "exp": int(time.time()) + 3600,
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="ES256")


class FakeWorker:
    def __init__(self) -> None:
        self.wakes = 0

    def wake(self) -> None:
        self.wakes += 1


class FakeRepository:
    def __init__(self, retry_result: str = "accepted") -> None:
        self.retry_result = retry_result
        self.retry_calls: list[tuple[str, str]] = []

    def request_retry(self, document_id, user_id):
        self.retry_calls.append((document_id, user_id))
        return self.retry_result


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret")
    get_settings.cache_clear()

    def build(repository=None):
        app = FastAPI()
        app.include_router(processing.router)
        app.state.worker = FakeWorker()
        app.state.repository = repository or FakeRepository()
        app.dependency_overrides[get_jwks_client] = FakeJwksClient
        return TestClient(app), app

    yield build
    get_settings.cache_clear()


def auth(value: str) -> dict:
    return {"Authorization": f"Bearer {value}"}


def test_health_needs_no_session(make_client):
    client, _ = make_client()

    assert client.get("/health").json() == {"status": "ok"}


def test_wake_with_a_valid_session_wakes_the_worker(make_client):
    client, app = make_client()

    response = client.post("/v1/processing/wake", headers=auth(token()))

    assert response.status_code == 202
    assert app.state.worker.wakes == 1


@pytest.mark.parametrize(
    "headers",
    [
        {},
        auth("not-a-jwt"),
        auth(token(key=OTHER_KEY)),
        auth(token(exp=int(time.time()) - 60)),
        auth(token(aud="anon")),
        auth(token(iss="https://other.supabase.co/auth/v1")),
        auth(token(role="anon")),
        auth(token(sub=None)),
    ],
    ids=["missing", "garbage", "other-signature", "expired", "audience", "issuer", "role", "no-sub"],
)
def test_requests_without_a_valid_session_are_rejected(make_client, headers):
    client, app = make_client()

    response = client.post("/v1/processing/wake", headers=headers)

    assert response.status_code == 401
    assert app.state.worker.wakes == 0


def test_retry_uses_the_user_from_the_token_and_wakes_the_worker(make_client):
    repository = FakeRepository("accepted")
    client, app = make_client(repository)

    response = client.post(f"/v1/documents/{DOCUMENT_ID}/retry", headers=auth(token()))

    assert response.status_code == 202
    assert repository.retry_calls == [(DOCUMENT_ID, USER_ID)]
    assert app.state.worker.wakes == 1


def test_retry_of_another_users_document_is_not_found(make_client):
    client, app = make_client(FakeRepository("not_found"))

    response = client.post(f"/v1/documents/{DOCUMENT_ID}/retry", headers=auth(token()))

    assert response.status_code == 404
    assert app.state.worker.wakes == 0


def test_retry_of_a_file_error_is_a_conflict(make_client):
    client, _ = make_client(FakeRepository("not_retryable"))

    response = client.post(f"/v1/documents/{DOCUMENT_ID}/retry", headers=auth(token()))

    assert response.status_code == 409


def test_retry_needs_a_session_and_a_valid_id(make_client):
    repository = FakeRepository()
    client, _ = make_client(repository)

    assert client.post(f"/v1/documents/{DOCUMENT_ID}/retry").status_code == 401
    assert client.post("/v1/documents/not-a-uuid/retry", headers=auth(token())).status_code == 422
    assert repository.retry_calls == []


def test_random_document_ids_are_passed_as_text(make_client):
    repository = FakeRepository()
    client, _ = make_client(repository)
    document_id = str(uuid.uuid4())

    client.post(f"/v1/documents/{document_id}/retry", headers=auth(token()))

    assert repository.retry_calls[0][0] == document_id

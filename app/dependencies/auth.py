"""Supabase session check for protected endpoints (SPEC RF-08).

Supabase Auth signs access tokens with an asymmetric key (ES256), so they are verified
locally with the project's public JWKS, without calling Supabase on every request.
"""

from __future__ import annotations

from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import JWT_AUDIENCE, get_settings

ALGORITHMS = ["ES256", "RS256"]

_bearer = HTTPBearer(auto_error=False)


@lru_cache
def get_jwks_client() -> jwt.PyJWKClient:
    # Caches the keys; a new kid (key rotation) triggers one refetch.
    return jwt.PyJWKClient(f"{get_settings().supabase_url}/auth/v1/.well-known/jwks.json", cache_keys=True)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid_session",
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_access_token(token: str, jwks_client: jwt.PyJWKClient, issuer: str) -> str:
    """Returns the user id (sub) of a valid Supabase access token, or raises HTTPException(401)."""
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALGORITHMS,
            audience=JWT_AUDIENCE,
            issuer=issuer,
            options={"require": ["exp", "sub", "aud", "iss"]},
        )
    except jwt.PyJWTError as e:
        raise _unauthorized() from e
    if claims.get("role") != "authenticated" or not claims.get("sub"):
        raise _unauthorized()
    return str(claims["sub"])


def current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    jwks_client: jwt.PyJWKClient = Depends(get_jwks_client),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    issuer = f"{get_settings().supabase_url}/auth/v1"
    return verify_access_token(credentials.credentials, jwks_client, issuer)

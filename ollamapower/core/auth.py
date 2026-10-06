"""The shared control-plane token (DCP's ``DCP_TOKEN``, now ``OLLAMAPOWER_TOKEN``).

Every agent proves it belongs to this cluster by sending the token inside the
encrypted session. Tokens are 256-bit random values from ``secrets`` and are
compared in constant time; DCP's plain ``!=`` leaked timing.
"""

from __future__ import annotations

import hmac
import secrets

TOKEN_ENV = "OLLAMAPOWER_TOKEN"
TOKEN_PREFIX = "op_"
TOKEN_BYTES = 32  # 256 bits
# Hand-made tokens shorter than this are refused: they are guessable.
MIN_TOKEN_CHARS = 32


class TokenError(ValueError):
    """The configured token is missing or too weak."""


def new_token() -> str:
    """A fresh token, e.g. for ``ollamapower init``."""
    return TOKEN_PREFIX + secrets.token_urlsafe(TOKEN_BYTES)


def token_weakness(token: str | None) -> str | None:
    """Why ``token`` can't be used ("is not set", "is too short ..."), or None if it is strong enough."""
    if not token:
        return "is not set"
    if len(token) < MIN_TOKEN_CHARS:
        return f"is too short (at least {MIN_TOKEN_CHARS} characters)"
    return None


def validate_token(token: str | None) -> str:
    """Return ``token`` if it is strong enough to use, else raise ``TokenError``."""
    weakness = token_weakness(token)
    if not token:
        raise TokenError(f"{TOKEN_ENV} is not set; run `ollamapower init` to create one")
    if weakness:
        raise TokenError(f"{TOKEN_ENV} {weakness}; run `ollamapower init`")
    return token


def check_token(given: str | None, expected: str | None) -> bool:
    """Constant-time comparison; an empty expected token never matches."""
    if not expected or not isinstance(given, str):
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))

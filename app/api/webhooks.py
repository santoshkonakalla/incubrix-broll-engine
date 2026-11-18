"""HMAC-SHA256 signed webhook delivery (Integration Standard #3)."""

import hashlib
import hmac
import json
from typing import Any, Dict

import httpx

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("webhooks")


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(payload_bytes: bytes, signature_header: str, secret: str) -> bool:
    expected = sign_payload(payload_bytes, secret)
    return hmac.compare_digest(expected, signature_header)


def send_webhook(callback_url: str, payload: Dict[str, Any], secret: str | None = None) -> bool:
    settings = get_settings()
    signing_secret = secret if secret is not None else settings.webhook_hmac_secret
    body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    signature = sign_payload(body, signing_secret)

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                callback_url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-IncuBrix-Signature": signature,
                },
            )
            logger.info(
                "webhook_delivered",
                callback_url=callback_url,
                status_code=resp.status_code,
            )
            return 200 <= resp.status_code < 300
    except Exception as exc:
        logger.warning("webhook_delivery_failed", callback_url=callback_url, error=str(exc))
        return False

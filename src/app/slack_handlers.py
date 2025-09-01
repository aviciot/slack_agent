
import hmac, hashlib, time, os
from typing import Dict

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")

def verify_slack_signature(headers: Dict[str, str], body: bytes) -> bool:
    if not SLACK_SIGNING_SECRET:
        return True  # Dev mode: skip if not set
    timestamp = headers.get("X-Slack-Request-Timestamp")
    sig = headers.get("X-Slack-Signature", "")
    if not timestamp or not sig:
        return False
    # Prevent replay
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    h = hmac.new(SLACK_SIGNING_SECRET.encode(), basestring.encode(), hashlib.sha256).hexdigest()
    expected = f"v0={h}"
    return hmac.compare_digest(expected, sig)

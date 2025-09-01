import logging
import json
import sys
import os
import uuid
import contextvars
from datetime import datetime

# Context var to hold request_id across functions
_request_id_ctx = contextvars.ContextVar("request_id", default=None)

def set_request_id(rid: str = None) -> str:
    """Generate or set request_id for current context."""
    if rid is None:
        rid = str(uuid.uuid4())
    _request_id_ctx.set(rid)
    return rid

def get_request_id() -> str:
    return _request_id_ctx.get()

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
            "logger": record.name,
        }
        if record.args and isinstance(record.args, dict):
            log_obj.update(record.args)
        return json.dumps(log_obj)

def configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json")

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers = []
    root.addHandler(handler)

    return root

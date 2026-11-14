"""Shared helper for polite requests to SEC EDGAR.

The SEC requires every automated request to identify who is making it
(a name and an email in the User-Agent header). We read that from the
EDGAR_IDENTITY environment variable so it is never written into the code.

    export EDGAR_IDENTITY="Your Name your.email@example.com"

The SEC also limits clients to 10 requests per second; we stay well under it.
"""

import os
import sys
import time

import requests

_MIN_SECONDS_BETWEEN_REQUESTS = 0.2
_last_request_time = 0.0


def session() -> requests.Session:
    identity = os.environ.get("EDGAR_IDENTITY", "").strip()
    if "@" not in identity:
        sys.exit(
            "Set your SEC identity first, e.g.\n"
            '  export EDGAR_IDENTITY="Your Name your.email@example.com"'
        )
    s = requests.Session()
    s.headers.update({"User-Agent": identity, "Accept-Encoding": "gzip, deflate"})
    return s


def get(s: requests.Session, url: str, **kwargs) -> requests.Response:
    """GET with a small delay between calls and a few retries."""
    global _last_request_time
    for attempt in range(4):
        wait = _MIN_SECONDS_BETWEEN_REQUESTS - (time.time() - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.time()
        r = s.get(url, timeout=120, **kwargs)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 503):
            time.sleep(2 ** attempt * 5)  # back off and retry
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r

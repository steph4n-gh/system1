"""Serve the unqualified development artifact for n8n integration checks.

This is not the qualified takeover demonstration. The endpoint never calls a
teacher; the workflow is responsible for explicit review/fallback handling.
"""
import argparse
import os
from pathlib import Path

from threadpoolctl import threadpool_limits
import uvicorn

from develop import OUTPUT
from runtime_probe import LocalCandidate
from system1.integrations.inbox_zero import InboxZeroApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=OUTPUT / "clinc-runtime-candidate")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    candidate = LocalCandidate(args.candidate)
    # Reuse the existing authenticated, size-bounded JSON ASGI transport.
    app = InboxZeroApp(candidate, os.environ.get("SYSTEM1_CLASSIFIER_TOKEN", ""))
    with threadpool_limits(limits=1):
        uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False,
                    limit_concurrency=16, timeout_keep_alive=5)


if __name__ == "__main__":
    main()

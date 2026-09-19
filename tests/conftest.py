"""Tests use explicit HTTP mocks, never the live paid TypeSafe endpoint."""
import urllib.error
import urllib.request

import pytest


@pytest.fixture(autouse=True)
def block_live_typesafe_http(monkeypatch):
    original = urllib.request.urlopen

    def urlopen(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if 'api.typesafe.ai' in url:
            raise urllib.error.URLError('Live TypeSafe HTTP is disabled in tests; install a response mock')
        return original(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

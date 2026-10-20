"""
tests/known_bugs.py — Marker for defects that are documented but not fixed yet.

A test decorated with ``known_bug`` asserts the *correct* behaviour and is
expected to fail today. ``strict=True`` turns an unexpected pass into a test
failure, so the commit that fixes a bug must also delete its marker: the
suite cannot silently drift out of sync with the code.
"""

from __future__ import annotations

import pytest


def known_bug(
    reason: str,
    raises: type[BaseException] | tuple[type[BaseException], ...] | None = None,
) -> pytest.MarkDecorator:
    """
    Strict xfail with a uniform reason prefix.

    *raises* narrows which failure counts as "the known bug"; any other
    exception is reported as a real failure instead of being absorbed.
    """
    return pytest.mark.xfail(strict=True, reason=f"known bug: {reason}", raises=raises)

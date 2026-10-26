"""Nettoyage du texte avant vectorisation."""

from __future__ import annotations

import html
import re

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEATED_CHAR = re.compile(r"(.)\1{2,}")
_NON_TEXT = re.compile(r"[^a-z0-9'!?\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise une critique : HTML, URLs, casse, ponctuation et espaces.

    Les ``!`` et ``?`` sont conservés car ils portent une partie du sentiment,
    et les négations (``don't``, ``isn't``) gardent leur apostrophe.
    """
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _REPEATED_CHAR.sub(r"\1\1", text)
    text = _NON_TEXT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()

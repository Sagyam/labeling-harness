"""A provider backed by a fixed table, for tests and offline demos."""

from __future__ import annotations

from app.translit.base import TranslitProvider


class StaticTranslitProvider(TranslitProvider):
    """Returns candidates from a dictionary supplied at construction."""

    name = "static"

    def __init__(self, table: dict[str, list[str]] | None = None) -> None:
        self.table = {k.lower(): v for k, v in (table or {}).items()}

    def suggest(self, latin_token: str) -> list[str]:
        """Look the token up in the table, returning an empty list when it is absent."""
        return list(self.table.get(latin_token.lower().strip(), []))

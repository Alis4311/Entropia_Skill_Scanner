from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Literal, Sequence, Tuple

from entropia_skillscanner.core import SkillRow
from entropia_skillscanner.skill_dedupe import merge_skill_rows


FieldName = Literal["rows", "status", "warnings"]
Subscriber = Callable[[object], None]


class SkillScannerViewModel:
    """
    Observable view-model for the Tk UI.

    Exposes three pieces of state:
      - rows: List[SkillRow]
      - status: str
      - warnings: Tuple[str, ...]
    Subscribers can listen to individual fields and receive updates when values change.
    """

    def __init__(self) -> None:
        self.rows: List[SkillRow] = []
        self.status: str = ""
        self.warnings: Tuple[str, ...] = ()
        self._subscribers: Dict[FieldName, List[Subscriber]] = {
            "rows": [],
            "status": [],
            "warnings": [],
        }

    def subscribe(self, field: FieldName, fn: Subscriber) -> Callable[[], None]:
        """
        Subscribe to a field; returns an unsubscribe callable.
        Invokes the callback immediately with the current value.
        """
        if field not in self._subscribers:
            raise ValueError(f"Unknown field '{field}'")

        self._subscribers[field].append(fn)
        fn(self._get_value(field))

        def unsubscribe() -> None:
            try:
                self._subscribers[field].remove(fn)
            except ValueError:
                pass

        return unsubscribe

    # ---- mutations ----

    def set_status(self, status: str) -> None:
        if status == self.status:
            return
        self.status = status
        self._notify("status")

    def set_rows(self, rows: Iterable[SkillRow]) -> None:
        self.rows = merge_skill_rows([], rows)
        self._notify("rows")

    def append_rows(self, new_rows: Iterable[SkillRow]) -> None:
        if not new_rows:
            return
        self.rows = merge_skill_rows(self.rows, new_rows)
        self._notify("rows")

    def set_warnings(self, warnings: Sequence[str]) -> None:
        warnings_tuple = tuple(warnings)
        if warnings_tuple == self.warnings:
            return
        self.warnings = warnings_tuple
        self._notify("warnings")

    # ---- internal ----

    def _notify(self, field: FieldName) -> None:
        value = self._get_value(field)
        for fn in list(self._subscribers[field]):
            fn(value)

    def _get_value(self, field: FieldName):
        if field == "rows":
            return self.rows
        if field == "status":
            return self.status
        if field == "warnings":
            return self.warnings
        raise ValueError(f"Unknown field '{field}'")

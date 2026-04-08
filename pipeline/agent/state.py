"""
Assignment state management for the agentic 10-K itemization pipeline.

Append-only event log pattern. All data types are frozen/immutable.
The AssignmentState class is the sole mutable container, but it only
grows (events are never deleted) and assignments are replaced atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from pipeline.config import ITEM_SEQ_INDEX


# ---------------------------------------------------------------------------
# Immutable data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemAssignment:
    """A single item pinned to a character position via an anchor."""

    item_name: str
    anchor_id: str
    char_position: int
    reasoning: str


@dataclass(frozen=True)
class StateEvent:
    """One entry in the append-only event log."""

    turn: int
    action: str  # "assign" | "unassign"
    item_name: str
    anchor_id: str | None
    char_position: int | None
    reasoning: str


class ItemBoundary(NamedTuple):
    """Lightweight boundary tuple consumed by the slicer."""

    item_name: str
    char_position: int


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class AssignmentState:
    """Manages item assignments with an append-only event log.

    * ``assign_item`` creates (or overwrites) an assignment.
    * ``unassign_item`` removes an assignment.
    * ``get_assignments`` returns assignments sorted by char_position.
    * ``get_event_log`` returns the full history.
    * ``to_boundaries`` converts current assignments to ``ItemBoundary`` list.
    """

    def __init__(self) -> None:
        self._assignments: dict[str, ItemAssignment] = {}
        self._events: list[StateEvent] = []
        self._turn: int = 0

    # -- mutators (each records an event) -----------------------------------

    def assign_item(
        self,
        item_name: str,
        anchor_id: str,
        char_position: int,
        reasoning: str = "",
    ) -> ItemAssignment:
        """Create or overwrite an assignment for *item_name*.

        Returns the newly created ``ItemAssignment``.
        """
        assignment = ItemAssignment(
            item_name=item_name,
            anchor_id=anchor_id,
            char_position=char_position,
            reasoning=reasoning,
        )
        self._turn += 1
        self._events.append(
            StateEvent(
                turn=self._turn,
                action="assign",
                item_name=item_name,
                anchor_id=anchor_id,
                char_position=char_position,
                reasoning=reasoning,
            )
        )
        self._assignments = {**self._assignments, item_name: assignment}
        return assignment

    def unassign_item(self, item_name: str, reasoning: str = "") -> bool:
        """Remove the assignment for *item_name*.

        Returns ``True`` if the item was previously assigned, ``False``
        otherwise.  An event is always recorded.
        """
        was_assigned = item_name in self._assignments
        self._turn += 1
        self._events.append(
            StateEvent(
                turn=self._turn,
                action="unassign",
                item_name=item_name,
                anchor_id=None,
                char_position=None,
                reasoning=reasoning,
            )
        )
        if was_assigned:
            self._assignments = {
                k: v for k, v in self._assignments.items() if k != item_name
            }
        return was_assigned

    # -- queries ------------------------------------------------------------

    def get_assignments(self) -> list[ItemAssignment]:
        """Return current assignments sorted by ``char_position``."""
        return sorted(
            self._assignments.values(),
            key=lambda a: (a.char_position if a.char_position is not None else -1),
        )

    def get_assignment(self, item_name: str) -> ItemAssignment | None:
        """Return the assignment for a single item, or ``None``."""
        return self._assignments.get(item_name)

    def get_event_log(self) -> list[StateEvent]:
        """Return a copy of the full event history."""
        return list(self._events)

    @property
    def assigned_count(self) -> int:
        return len(self._assignments)

    @property
    def turn(self) -> int:
        return self._turn

    # -- conversion ---------------------------------------------------------

    def to_boundaries(self) -> list[ItemBoundary]:
        """Convert current assignments to a sorted list of ``ItemBoundary``.

        The list is ordered by ``char_position`` (ties broken by canonical
        item order) so it can be fed directly to the slicer.
        """
        return [
            ItemBoundary(
                item_name=a.item_name,
                char_position=a.char_position,
            )
            for a in sorted(
                self._assignments.values(),
                key=lambda a: (
                    a.char_position if a.char_position is not None else -1,
                    ITEM_SEQ_INDEX.get(a.item_name, 999),
                ),
            )
        ]

"""
Chess helpers for dataset generation and trace building.

This module is intentionally small and stable:
- Parse movetext (SAN) from common Lichess exports
- Enumerate legal moves in UCI form
"""

from __future__ import annotations

import re
from typing import List

import chess


def parse_movetext(movetext: str) -> List[str]:
    """
    Parse a Lichess movetext string into SAN moves.

    Supports common Lichess PGN-like movetext formats, including clock blocks:
        "1. e4 { [%clk 0:05:00] } e5 2. Nf3 Nc6 3. Bb5 a6 1-0"

    Returns:
        A list of SAN moves in order (e.g. ["e4", "e5", "Nf3", ...]).
    """

    if not movetext:
        return []

    cleaned = re.sub(r"\{[^}]*\}", "", movetext)  # {...} annotations
    cleaned = re.sub(r"\d+\.+\s*", "", cleaned)  # "1." / "1..." prefixes
    cleaned = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", "", cleaned)  # results
    cleaned = re.sub(r"\?+|\!+", "", cleaned)  # simple punctuation nags

    return [token.strip() for token in cleaned.split() if token.strip()]


def get_legal_moves_uci(board: chess.Board) -> str:
    """Return all legal moves as a space-separated UCI string."""

    return " ".join(move.uci() for move in board.legal_moves)

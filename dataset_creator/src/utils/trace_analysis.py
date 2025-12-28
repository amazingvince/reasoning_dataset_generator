"""
Analysis helpers for reasoning-trace generation.

This module converts Stockfish evaluations stored in a position dict into a
lightweight `PositionAnalysis` structure that `ReasoningTraceGenerator` can
consume to build a natural-language trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chess

from src.utils.chess_utils import get_legal_moves_uci


@dataclass
class MoveAnalysis:
    uci: str
    san: str
    centipawn: int
    cp_loss: int
    category: str
    mate_in: Optional[int] = None
    win_probability: float = 0.5
    pv_uci: List[str] = field(default_factory=list)


@dataclass
class PositionAnalysis:
    fen: str
    move_analyses: List[MoveAnalysis]
    best_move_uci: str
    best_move_san: str
    best_score_cp: int

    best_pv: List[str] = field(default_factory=list)
    move_probs: Dict[str, float] = field(default_factory=dict)
    top_k_moves: List[str] = field(default_factory=list)

    shallow_move_cps: Dict[str, int] = field(default_factory=dict)
    shallow_move_win_probs: Dict[str, float] = field(default_factory=dict)
    confirm_move_cps: Dict[str, int] = field(default_factory=dict)
    confirm_move_win_probs: Dict[str, float] = field(default_factory=dict)


def categorize_move(cp_loss: int, mate_in: Optional[int] = None) -> str:
    if mate_in is not None:
        if mate_in > 0:
            return "winning mate"
        return "blunder"

    if cp_loss == 0:
        return "best move"
    if cp_loss <= 10:
        return "excellent"
    if cp_loss <= 30:
        return "good"
    if cp_loss <= 50:
        return "slight inaccuracy"
    if cp_loss <= 100:
        return "inaccuracy"
    if cp_loss <= 200:
        return "mistake"
    return "blunder"


def cp_to_win_probability(cp: int) -> float:
    return 1.0 / (1.0 + 10 ** (-cp / 400.0))


def cp_to_probability_distribution(
    move_cps: Dict[str, int],
    all_legal_moves: List[str],
    *,
    temperature: float = 100.0,
    min_probability: float = 0.001,
) -> Dict[str, float]:
    num_moves = len(all_legal_moves)
    if num_moves == 0:
        return {}

    reserved_mass = float(min_probability) * num_moves
    if reserved_mass >= 1.0:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    remaining_mass = 1.0 - reserved_mass

    if not move_cps:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    temperature = float(temperature)
    temperature = max(1e-6, temperature)

    moves = list(move_cps.keys())
    cps = [int(move_cps[m]) for m in moves]
    max_cp = max(cps)

    exp_scores = [math.exp((cp - max_cp) / temperature) for cp in cps]
    denom = sum(exp_scores)
    if denom <= 0:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    probs = {m: float(min_probability) for m in all_legal_moves}
    for move, exp_score in zip(moves, exp_scores):
        probs[move] = probs.get(move, float(min_probability)) + (exp_score / denom) * remaining_mass

    total = sum(probs.values())
    if total <= 0:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    return {m: p / total for m, p in probs.items()}


def position_with_eval_to_analysis(
    position: Dict[str, Any],
    *,
    min_probability: float = 0.001,
    stockfish_temperature: float = 100.0,
) -> PositionAnalysis:
    fen = position.get("fen")
    if not fen:
        raise ValueError("position must contain 'fen'")

    board = chess.Board(fen)
    all_legal_moves = position.get("legal_moves_uci") or get_legal_moves_uci(board)
    all_legal_moves_list = [m for m in str(all_legal_moves).split() if m]

    move_evaluations = position.get("move_evaluations") or []
    best_move_uci = position.get("best_move_uci") or ""
    best_move_san = position.get("best_move_san") or ""
    best_score_cp = int(position.get("best_score_cp") or 0)

    move_cps: Dict[str, int] = {}
    move_analyses: List[MoveAnalysis] = []
    for mv in move_evaluations:
        uci = mv.get("uci")
        if not uci:
            continue
        cp = int(mv.get("centipawn") or 0)
        move_cps[str(uci)] = cp

        mate_in = mv.get("mate_in")
        mate_in = int(mate_in) if mate_in is not None else None
        cp_loss = best_score_cp - cp

        pv_uci = mv.get("pv_uci") or []
        if isinstance(pv_uci, str):
            pv_uci = [p for p in pv_uci.split() if p]

        move_analyses.append(
            MoveAnalysis(
                uci=str(uci),
                san=str(mv.get("san") or uci),
                centipawn=cp,
                cp_loss=cp_loss,
                category=categorize_move(cp_loss, mate_in),
                mate_in=mate_in,
                win_probability=cp_to_win_probability(cp),
                pv_uci=[str(p) for p in pv_uci if p],
            )
        )

    move_analyses.sort(key=lambda m: m.centipawn, reverse=True)

    if not best_move_uci and move_analyses:
        best_move_uci = move_analyses[0].uci
        best_move_san = move_analyses[0].san
        best_score_cp = move_analyses[0].centipawn

    move_probs = cp_to_probability_distribution(
        move_cps=move_cps,
        all_legal_moves=all_legal_moves_list,
        temperature=float(stockfish_temperature),
        min_probability=float(min_probability),
    )

    def _parse_cp_map(value: Any) -> Dict[str, int]:
        if not value:
            return {}
        if not isinstance(value, dict):
            return {}
        out: Dict[str, int] = {}
        for key, cp in value.items():
            if key is None:
                continue
            try:
                out[str(key)] = int(cp)
            except (TypeError, ValueError):
                continue
        return out

    shallow_move_cps = _parse_cp_map(position.get("shallow_move_cps"))
    confirm_move_cps = _parse_cp_map(position.get("confirm_move_cps"))
    shallow_move_win_probs = {uci: cp_to_win_probability(cp) for uci, cp in shallow_move_cps.items()}
    confirm_move_win_probs = {uci: cp_to_win_probability(cp) for uci, cp in confirm_move_cps.items()}

    return PositionAnalysis(
        fen=fen,
        move_analyses=move_analyses,
        best_move_uci=best_move_uci,
        best_move_san=best_move_san,
        best_score_cp=best_score_cp,
        best_pv=[],
        move_probs=move_probs,
        top_k_moves=[ma.uci for ma in move_analyses[:5]],
        shallow_move_cps=shallow_move_cps,
        shallow_move_win_probs=shallow_move_win_probs,
        confirm_move_cps=confirm_move_cps,
        confirm_move_win_probs=confirm_move_win_probs,
    )

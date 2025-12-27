"""
Reasoning trace generator for chess datasets.

Creates human-style analysis text using Stockfish move analysis, opening
lookups, and optional tablebase probes. Output stays in UCI notation by
default to align with move tokens.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess

try:
    from ..utils.chess_utils import parse_movetext
except ImportError:
    from src.utils.chess_utils import parse_movetext


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

WEAK_SQUARE_CANDIDATES = [
    chess.D4, chess.E4, chess.D5, chess.E5,
    chess.C4, chess.F4, chess.C5, chess.F5,
]

# Style configuration for trace generation variety
STYLE_CONFIGS = {
    "thorough": {
        "default_weight": 0.40,
        "target_length": "long",
    },
    "concise": {
        "default_weight": 0.15,
        "target_length": "medium",
    },
    "tactical": {
        "default_weight": 0.10,
        "target_length": "medium",
    },
    "quick": {
        "default_weight": 0.10,
        "target_length": "short",
    },
    "problem_focused": {
        "default_weight": 0.10,
        "target_length": "medium",
    },
    "intuition": {
        "default_weight": 0.08,
        "target_length": "medium",
    },
    "comparison_focused": {
        "default_weight": 0.07,
        "target_length": "medium",
    },
    "dialogue": {
        "default_weight": 0.0,
        "target_length": "medium",
    },
}

# Expanded phrase library for variety in trace generation
PHRASE_LIBRARY = {
    "orientation_starters": [
        "Let me look at this position.",
        "First impression:",
        "At a glance:",
        "What's going on here?",
        "Looking at the board:",
        "Examining this position:",
        "Starting with the basics:",
        "What do we have here?",
        "Taking stock of the position:",
        "Let me assess this:",
    ],
    "candidate_intros": [
        "Moves to consider:",
        "Candidates:",
        "The main candidates are:",
        "A few moves catch my eye:",
        "The candidates that stand out are:",
        "Several options here:",
        "What deserves attention:",
        "The moves worth checking:",
        "Options on the table:",
        "Let me consider:",
    ],
    "analysis_starters": [
        "Looking at {move},",
        "For {move},",
        "Considering {move},",
        "What about {move}?",
        "Does {move} work here?",
        "How about {move}?",
        "Checking {move}:",
        "The idea with {move}:",
        "If we try {move},",
        "Examining {move}:",
    ],
    "uncertainty_phrases": [
        "I'm not entirely sure, but",
        "This seems to",
        "My instinct says",
        "This feels like",
        "It looks like",
        "Probably",
        "I think",
        "This might",
        "Hard to say for certain, but",
        "It appears that",
    ],
    "reconsideration_phrases": [
        "Actually, looking again...",
        "Wait, I missed...",
        "Hmm, but...",
        "On second thought...",
        "That changes things...",
        "I need to reconsider...",
        "Hold on...",
        "Actually, what about...",
        "Let me think again...",
        "Reconsidering this...",
    ],
    "dead_end_phrases": [
        "This doesn't work because",
        "I looked at {move} but it fails to",
        "Unfortunately, this runs into",
        "This would be nice, but",
        "The problem is",
        "{move} looked promising but",
        "At first I liked {move}, but",
        "This falls short because",
        "Doesn't quite get there.",
        "Almost, but not quite.",
        "Close but no cigar.",
        "{move} has issues.",
        "Can't make it work.",
        "Tried this, didn't pan out.",
        "Not as good as it looks.",
    ],
    "comparison_phrases": [
        "Compared to {move}",
        "Unlike {move}, this",
        "The difference is",
        "While {move1} does X, {move2} achieves Y",
        "Both moves have merit, but",
        "Weighing {move1} against {move2}",
        "The trade-off is",
        "Between these options,",
    ],
    "conclusion_intros": [
        "After considering everything,",
        "Taking this all into account,",
        "Given these factors,",
        "Based on this analysis,",
        "Having explored the options,",
        "The analysis points to",
        "All things considered,",
        "Weighing everything up,",
        "After working through this,",
        "Looking at the full picture,",
        "This leads me to",
        "The conclusion is clear:",
        "Everything points to",
        "My choice here is",
        "The right move is",
    ],
    "conclusion_formats": [
        "{move} is the move.",
        "{move} is the answer.",
        "{move} stands out as best.",
        "I'm playing {move}.",
        "going with {move}.",
        "{move} is what I'd play.",
        "{move} gets the nod.",
        "{move} is the call.",
        "it has to be {move}.",
        "{move} is the pick.",
        "{move} is clearly best.",
        "the move is {move}.",
        "{move} wins out.",
        "{move} is the right choice.",
        "{move} makes the most sense.",
    ],
    "quality_excellent": [
        "This looks very strong.",
        "Clearly the best option.",
        "This is the move.",
        "Definitely the right choice.",
        "This stands out.",
        "Excellent choice here.",
        "A clear winner.",
        "Nothing else comes close.",
        "The standout candidate.",
        "This has to be it.",
        "No real competition here.",
        "The obvious choice.",
    ],
    "quality_good": [
        "This seems promising.",
        "Looks good.",
        "A solid choice.",
        "This works well.",
        "Reasonable move.",
        "This looks fine.",
        "Sensible option.",
        "Can't go wrong here.",
        "A natural move.",
        "This holds up.",
        "Reliable choice.",
        "Makes sense.",
    ],
    "quality_playable": [
        "About equal.",
        "Playable.",
        "Reasonable.",
        "This keeps the balance.",
        "Not bad.",
        "Acceptable.",
        "Nothing wrong with this.",
        "Fair enough.",
        "Decent.",
        "Can be considered.",
        "Within the normal range.",
        "OK move.",
    ],
    "quality_dubious": [
        "Not so clear.",
        "Has some problems.",
        "Risky.",
        "Questionable.",
        "This seems off.",
        "Not convincing.",
        "A bit suspect.",
        "Probably inaccurate.",
        "May not hold up.",
        "There are issues.",
        "Looks shaky.",
        "Second best at most.",
    ],
    "quick_ideas": [
        "active",
        "developing",
        "central",
        "flexible",
        "solid",
        "natural",
        "safe",
        "dynamic",
        "aggressive",
        "positional",
        "prophylactic",
        "tempo",
        "space",
        "coordination",
        "pressure",
    ],
    "problem_intros": [
        "The challenge here:",
        "The main problem is",
        "What needs solving?",
        "The key question:",
        "The issue is",
        "What's the task?",
    ],
    "failed_attempt_intros": [
        "First thought: {move}.",
        "What about {move}?",
        "Trying {move} first.",
        "My first instinct: {move}.",
        "Initial idea: {move}.",
    ],
    "failed_attempt_second": [
        "Second attempt: {move}.",
        "Maybe {move} instead?",
        "How about {move}?",
        "Another option: {move}.",
        "Alternatively, {move}.",
    ],
    "failure_reasons_bad": [
        "But this loses material after a tactical shot.",
        "This runs into a strong reply and falls apart.",
        "Unfortunately this has a concrete refutation.",
        "But there's a problem with this.",
        "This doesn't quite work.",
    ],
    "failure_reasons_ok": [
        "This is playable but not convincing.",
        "It works but doesn't solve the real problem.",
        "Close, but there's something better.",
        "Decent, but not optimal.",
        "Not bad, but we can improve.",
    ],
    "failure_reasons_close": [
        "This is decent but not the sharpest.",
        "Reasonable, but we can do better.",
        "Solid, though not optimal.",
        "Fine, but there's more.",
        "Good, but not quite right.",
    ],
    "solution_intros": [
        "The solution: {move}.",
        "Here's what works: {move}.",
        "Finally, {move} does the job.",
        "The answer is {move}.",
        "What works: {move}.",
        "{move} solves it.",
    ],
    "intuition_first_impressions": [
        "Gut feeling: this position has some tension.",
        "My instinct says there should be something active here.",
        "First impression: this looks like a key moment.",
        "This feels like a critical position.",
        "Something important is happening here.",
        "Instinct tells me this matters.",
    ],
    "intuition_preference_intros": [
        "The move that jumps out is {move}. Let me verify this.",
        "My first instinct says {move}. But let's check the details.",
        "{move} catches my eye. Time to see if it holds up.",
        "Initially drawn to {move}. Let me confirm.",
        "{move} seems right. Worth checking.",
    ],
    "intuition_verify": [
        "Checking {move} concretely...",
        "Let me verify {move}.",
        "Running through {move} in my head...",
        "Testing {move}...",
        "Looking at {move} more closely...",
    ],
    "intuition_confirmed": [
        "Intuition confirmed. {move} is the right call.",
        "The verification backs up the gut feeling. {move} it is.",
        "My first instinct was right. Playing {move}.",
        "Yes, {move} holds up.",
        "Confirmed: {move} works.",
    ],
    "intuition_adjusted": [
        "Actually, after checking, {move} is better than I first thought.",
        "Adjusting my initial read. {move} is the move.",
        "The concrete lines favor {move} over my first choice.",
        "Changed my mind. {move} is right.",
        "On reflection, {move} is stronger.",
    ],
    "comparison_frames_three": [
        "The question: {m1}, {m2}, or {m3}?",
        "Three main options: {m1}, {m2}, {m3}. Which one?",
        "Deciding between {m1}, {m2}, and {m3}.",
        "Main candidates: {m1}, {m2}, {m3}.",
    ],
    "comparison_frames_two": [
        "Two main choices: {m1} or {m2}.",
        "The decision comes down to {m1} versus {m2}.",
        "Which is better: {m1} or {m2}?",
        "{m1} vs {m2}.",
    ],
    "comparison_pros_strong": [
        "strong initiative",
        "good chances",
        "active",
        "aggressive",
        "forcing",
    ],
    "comparison_pros_solid": [
        "solid",
        "safe",
        "flexible",
        "reliable",
        "sound",
    ],
    "comparison_pros_fighting": [
        "fighting",
        "creates complications",
        "ambitious",
        "interesting",
        "dynamic",
    ],
    "comparison_cons_best": [
        "hard to see downsides",
        "minor cons at most",
        "looks clean",
        "no real problems",
    ],
    "comparison_cons_inferior": [
        "less accurate",
        "misses the point",
        "not quite right",
        "has issues",
    ],
    "comparison_cons_close": [
        "slightly inferior",
        "second best",
        "close but not optimal",
        "nearly as good",
    ],
    "comparison_cons_marginal": [
        "very close call",
        "nearly equivalent",
        "marginal difference",
        "hard to distinguish",
    ],
    "comparison_key_difference": [
        "What makes {move} better: it's more forcing.",
        "The key: {move} wins material or forces a favorable exchange.",
        "{move} gives a clear edge that the others don't match.",
        "{move} is more forcing and keeps the initiative.",
        "The precision of {move} edges out the alternatives.",
        "{move} solves more problems at once.",
        "{move} is simply more accurate.",
    ],
    "comparison_conclusions": [
        "The choice: {move}.",
        "Going with {move}.",
        "{move} wins the comparison.",
        "Picking {move}.",
        "{move} is the one.",
    ],
    # Move-type specific phrases
    "capture_phrases": [
        "Winning material.",
        "Captures and gains.",
        "Takes the piece.",
        "Material gain.",
        "A good exchange.",
        "Picks up material.",
        "Cashes in.",
        "Wins the piece.",
        "Collects material.",
        "A profitable trade.",
        "Clears the way.",
        "Removes a defender.",
    ],
    "quiet_move_phrases": [
        "Improving the position.",
        "Building up slowly.",
        "A patient move.",
        "Strengthening the setup.",
        "No hurry.",
        "Preparing something.",
        "Subtle improvement.",
        "Quiet but strong.",
        "Getting ready.",
        "A calm approach.",
        "Maneuvering.",
        "Repositioning.",
    ],
    "check_phrases": [
        "Check!",
        "Giving check.",
        "Forces the king to move.",
        "Attacks the king.",
        "With check.",
        "The king is hit.",
        "Harassing the king.",
        "Keeping pressure on the king.",
        "Driving the king.",
        "The king must respond.",
        "No time to breathe.",
        "Relentless pressure.",
    ],
    # Emotional/aesthetic phrases
    "beautiful_sacrifice": [
        "A brilliant sacrifice.",
        "A stunning piece sacrifice.",
        "An aesthetic sacrifice that opens lines.",
        "A gorgeous material investment.",
        "Sacrificing material for overwhelming compensation.",
        "A beautiful exchange sacrifice.",
        "Material for the initiative - a classic trade.",
        "A spectacular sacrifice.",
    ],
    "surprising_quiet_move": [
        "A quiet move when tactics were expected.",
        "Surprising simplicity.",
        "An unexpected calm response.",
        "A subtle but strong move.",
        "Quiet but devastating.",
        "Not the obvious choice, but the right one.",
        "A refined, quiet solution.",
        "Positionally perfect.",
    ],
    "clever_defense": [
        "A resourceful defense.",
        "A clever defensive idea.",
        "An ingenious defensive resource.",
        "Defensive brilliance.",
        "A surprisingly solid defense.",
        "Finding the only move that holds.",
        "A tricky defensive setup.",
        "Defending with precision.",
    ],
    "tempo_phrases": [
        "Gains tempo by attacking.",
        "Develops with a threat.",
        "Forcing the opponent to respond.",
        "No time to waste - attacks immediately.",
        "Developing while creating threats.",
        "A move with initiative.",
        "Active and aggressive.",
        "Keeps up the pressure.",
    ],
    "coordination_phrases": [
        "The pieces work together here.",
        "Good piece coordination.",
        "The pieces are harmoniously placed.",
        "Everything is connected.",
        "The army works as one.",
        "Pieces supporting each other.",
    ],
}


@dataclass
class OpeningInfo:
    """Opening metadata resolved for a position."""
    eco: str
    name: str
    main_line: str
    move_count: int


@dataclass
class TablebaseInfo:
    """Syzygy tablebase probe result summary."""
    result: str
    dtz: Optional[int]
    piece_count: int


class OpeningBook:
    """Best-effort opening lookup from pre-downloaded Lichess TSV files."""
    def __init__(self, paths: Iterable[Path]):
        self.paths = [Path(p) for p in paths if p]
        self._loaded = False
        self._fen_to_opening: Dict[str, OpeningInfo] = {}
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        self._ensure_loaded()
        return bool(self._fen_to_opening)

    def load_error(self) -> Optional[str]:
        self._ensure_loaded()
        return self._load_error

    def lookup(self, board: chess.Board) -> Optional[OpeningInfo]:
        self._ensure_loaded()
        if not self._fen_to_opening:
            return None
        key = self._normalize_fen(board)
        return self._fen_to_opening.get(key)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        existing = [p for p in self.paths if p.exists()]
        if not existing:
            self._load_error = "no opening files found"
            return
        for path in existing:
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline().strip().split("\t")
                col = {name: idx for idx, name in enumerate(header)}
                for line in handle:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    eco = self._get_col(parts, col, "eco") or ""
                    name = self._get_col(parts, col, "name") or ""
                    uci_moves = self._extract_moves(parts, col)
                    if not uci_moves:
                        continue
                    main_line = " ".join(uci_moves)
                    self._index_line(eco, name, uci_moves, main_line)
        except Exception as exc:
            self._load_error = f"failed to load {path}: {exc}"

    def _extract_moves(self, parts: List[str], col: Dict[str, int]) -> List[str]:
        uci_moves = self._get_col(parts, col, "uci")
        if uci_moves:
            return [m for m in uci_moves.split() if m]
        pgn_moves = self._get_col(parts, col, "pgn")
        if not pgn_moves:
            pgn_moves = self._get_col(parts, col, "moves")
        if not pgn_moves:
            return []
        san_moves = parse_movetext(pgn_moves)
        board = chess.Board()
        uci_seq = []
        for san in san_moves:
            try:
                move = board.parse_san(san)
            except Exception:
                break
            uci_seq.append(move.uci())
            board.push(move)
        return uci_seq

    @staticmethod
    def _get_col(parts: List[str], col: Dict[str, int], key: str) -> Optional[str]:
        idx = col.get(key)
        if idx is None or idx >= len(parts):
            return None
        return parts[idx]

    def _index_line(
        self,
        eco: str,
        name: str,
        uci_moves: List[str],
        main_line: str,
    ) -> None:
        board = chess.Board()
        for idx, move_uci in enumerate(uci_moves):
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            board.push(move)
            key = self._normalize_fen(board)
            existing = self._fen_to_opening.get(key)
            if existing is None or idx + 1 >= existing.move_count:
                self._fen_to_opening[key] = OpeningInfo(
                    eco=eco,
                    name=name,
                    main_line=main_line,
                    move_count=idx + 1,
                )

    @staticmethod
    def _normalize_fen(board: chess.Board) -> str:
        fen = board.fen()
        parts = fen.split(" ")
        if len(parts) < 4:
            return fen
        return " ".join(parts[:4])


class TablebaseProbe:
    """Best-effort Syzygy WDL/DTZ probe (optional dependency)."""
    def __init__(self, paths: Iterable[Path]):
        self.paths = [Path(p) for p in paths if p]
        self._tb = None
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def load_error(self) -> Optional[str]:
        self._ensure_loaded()
        return self._load_error

    def probe(self, board: chess.Board) -> Optional[TablebaseInfo]:
        if len(board.piece_map()) > 7:
            return None
        if not self._ensure_loaded():
            return None
        try:
            wdl = self._tb.probe_wdl(board)
            dtz = self._tb.probe_dtz(board)
        except Exception:
            return None
        result = "drawing"
        if wdl > 0:
            result = "winning"
        elif wdl < 0:
            result = "losing"
        return TablebaseInfo(result=result, dtz=dtz, piece_count=len(board.piece_map()))

    def _ensure_loaded(self) -> bool:
        if self._tb is not None:
            return True
        if not self.paths:
            return False
        try:
            import chess.syzygy
        except Exception as exc:
            self._load_error = f"syzygy unavailable: {exc}"
            return False
        existing = [p for p in self.paths if p.exists()]
        if not existing:
            self._load_error = "no tablebase path found"
            return False
        try:
            self._tb = chess.syzygy.open_tablebase(existing[0])
            for path in existing[1:]:
                self._tb.add_directory(path)
            return True
        except Exception as exc:
            self._load_error = f"failed to load tablebase: {exc}"
            return False


class ReasoningTraceGenerator:
    """
    Generate diverse, human-style reasoning traces for chess positions.

    The generator consumes:
    - `position`: a dict with at least `fen` (and optionally `source`, etc.)
    - `analysis`: an object with `move_analyses`, `best_move_uci`, and related fields

    Output is plain text intended to be stored directly as `reasoning_trace` in
    the streaming JSONL dataset.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or {}
        self.tokenizer = tokenizer
        self.seed = self.config.get("seed")
        self._rng = random.Random(self.seed)

        opening_paths = self.config.get("opening_paths")
        if isinstance(opening_paths, (str, Path)):
            opening_paths = [opening_paths]
        if not opening_paths:
            opening_paths = self._default_opening_paths()
        self._opening = OpeningBook(opening_paths)

        tablebase_paths = self.config.get("tablebase_paths")
        if isinstance(tablebase_paths, (str, Path)):
            tablebase_paths = [tablebase_paths]
        if not tablebase_paths:
            tablebase_paths = self._default_tablebase_paths()
        self._tablebase = TablebaseProbe(tablebase_paths)

    def status(self) -> Dict[str, Any]:
        opening_available = self._opening.is_available()
        tablebase_available = self._tablebase.is_available()
        return {
            "opening_available": opening_available,
            "opening_error": None if opening_available else self._opening.load_error(),
            "opening_paths": [str(p) for p in self._opening.paths],
            "tablebase_available": tablebase_available,
            "tablebase_error": None if tablebase_available else self._tablebase.load_error(),
            "tablebase_paths": [str(p) for p in self._tablebase.paths],
        }

    def generate(
        self,
        position: Dict[str, Any],
        analysis: Any,
        rng: Optional[random.Random] = None,
    ) -> str:
        """
        Build a single reasoning trace for a position.

        Args:
            position: Example dict containing at least `fen` (optionally `source`).
            analysis: Position analysis object (see `src/utils/trace_analysis.PositionAnalysis`).
            rng: Optional RNG override for deterministic generation.

        Returns:
            A natural-language reasoning trace (no `<think>` wrapper).
        """
        source = position.get("source")
        cfg = self._resolve_config(source)
        rng = rng or self._rng

        fen = position.get("fen")
        if not fen:
            return "No position provided."
        board = chess.Board(fen)

        best_move_uci = analysis.best_move_uci or ""
        best_analysis = self._find_move(analysis, best_move_uci)
        if best_analysis is None and analysis.move_analyses:
            best_analysis = analysis.move_analyses[0]
            best_move_uci = best_analysis.uci

        candidates = self._select_candidates(analysis, cfg, rng)
        if not candidates and best_analysis is not None:
            candidates = [best_analysis]

        decision_profile = self._decision_profile(analysis, best_analysis, cfg)

        style = self._select_style(cfg, source, rng, position=position)
        style = self._maybe_override_style(style, decision_profile, rng, cfg)
        notation = cfg.get("move_notation", "uci")

        opening_info = self._opening.lookup(board) if cfg.get("include_opening", True) else None
        tablebase_info = self._tablebase.probe(board) if cfg.get("include_tablebase", True) else None

        opening_plan = self._opening_plan_line(opening_info, style, rng, cfg)
        endgame_technique = self._endgame_technique_line(board, tablebase_info, style, rng, cfg)

        fen_walkthrough = ""
        if cfg.get("include_fen_walkthrough", False):
            prob = float(cfg.get("fen_walkthrough_prob", 0.15))
            if style == "quick":
                prob *= 0.25
            elif style == "concise":
                prob *= 0.5
            if rng.random() < min(max(prob, 0.0), 1.0):
                fen_walkthrough = self._fen_walkthrough_line(board, style, rng, cfg)

        context_hint = self._context_hint_line(board, position, analysis, best_analysis, style, rng, cfg)
        decision_line = self._decision_profile_line(board, position, best_analysis, decision_profile, style, rng, cfg)

        if style == "dialogue":
            text = self._generate_dialogue_style(
                board=board,
                position=position,
                candidates=candidates,
                best_analysis=best_analysis,
                analysis=analysis,
                decision_profile=decision_profile,
                opening_info=opening_info,
                tablebase_info=tablebase_info,
                opening_plan=opening_plan,
                fen_walkthrough=fen_walkthrough,
                context_hint=context_hint,
                decision_line=decision_line,
                endgame_technique=endgame_technique,
                rng=rng,
                cfg=cfg,
                notation=notation,
            )
            return self._apply_guardrails(text, tablebase_info, cfg)

        preamble_parts = [p for p in (fen_walkthrough, context_hint, decision_line) if p]
        if style in ("quick", "problem_focused", "intuition", "comparison_focused"):
            extra_preamble = [opening_plan, endgame_technique]
            avoid_line = self._avoid_move_line(board, analysis, best_analysis, style, rng, cfg, notation)
            extra_preamble.append(avoid_line)
            preamble_parts.extend(p for p in extra_preamble if p)
        preamble = "\n\n".join(preamble_parts)

        def maybe_prepend_preamble(text: str) -> str:
            if not preamble:
                return text
            combined = f"{preamble}\n\n{text}"
            max_tokens = cfg.get("max_trace_tokens")
            if max_tokens and self._count_tokens(combined) > max_tokens:
                return text
            return combined

        # Dispatch to style-specific generators
        if style == "quick":
            text = self._generate_quick_style(board, candidates, best_analysis, analysis, rng, cfg, notation)
            return self._apply_guardrails(maybe_prepend_preamble(text), tablebase_info, cfg)
        elif style == "problem_focused":
            text = self._generate_problem_focused_style(
                board, candidates, best_analysis, analysis, rng, cfg, notation
            )
            return self._apply_guardrails(maybe_prepend_preamble(text), tablebase_info, cfg)
        elif style == "intuition":
            text = self._generate_intuition_style(board, candidates, best_analysis, analysis, rng, cfg, notation)
            return self._apply_guardrails(maybe_prepend_preamble(text), tablebase_info, cfg)
        elif style == "comparison_focused":
            text = self._generate_comparison_style(board, candidates, best_analysis, analysis, rng, cfg, notation)
            return self._apply_guardrails(maybe_prepend_preamble(text), tablebase_info, cfg)

        # Continue with existing flow for thorough/concise/tactical styles
        orientation = self._orientation_line(
            board,
            opening_info,
            tablebase_info,
            source,
            style,
            rng,
            cfg,
        ) if cfg.get("include_orientation", True) else ""

        fen_section = fen_walkthrough if cfg.get("include_fen_walkthrough", False) else ""

        assessment = self._assessment_line(
            board,
            best_analysis,
            tablebase_info,
            style,
            rng,
            cfg,
        ) if cfg.get("include_assessment", True) else ""

        context_section = context_hint if bool(cfg.get("include_context_hints", True)) else ""
        decision_section = decision_line if bool(cfg.get("include_decision_profile", True)) else ""
        opening_plan_section = opening_plan if bool(cfg.get("include_opening_plan_hints", True)) else ""
        endgame_section = endgame_technique if bool(cfg.get("include_endgame_technique", True)) else ""

        static_eval = ""
        if cfg.get("include_static_eval_notes", False):
            prob = float(cfg.get("static_eval_prob", 0.25))
            if rng.random() < min(max(prob, 0.0), 1.0):
                static_eval = self._static_eval_line(board, style, rng, cfg)

        threat_scan = ""
        if cfg.get("include_threat_scan", True):
            prob = float(cfg.get("threat_scan_prob", 1.0))
            if style == "quick":
                prob *= 0.25
            elif style == "concise":
                prob *= 0.6
            if rng.random() < min(max(prob, 0.0), 1.0):
                threat_scan = self._threat_scan_line(
                    board,
                    style,
                    rng,
                    notation,
                    cfg,
                )

        plan_line = ""
        if cfg.get("include_plan", False):
            if rng.random() < float(cfg.get("plan_prob", 0.5)):
                plan_line = self._plan_line(
                    board,
                    best_analysis,
                    style,
                    rng,
                    cfg,
                )

        opponent_line = ""
        if cfg.get("include_opponent_perspective", False):
            if rng.random() < float(cfg.get("opponent_perspective_prob", 0.4)):
                opponent_line = self._opponent_perspective_line(
                    board,
                    style,
                    rng,
                    notation,
                    cfg,
                )

        candidates_line = self._candidate_line(candidates, style, rng, notation)
        avoid_section = self._avoid_move_line(board, analysis, best_analysis, style, rng, cfg, notation)

        exploration_paragraphs = self._exploration_paragraphs(
            board,
            candidates,
            best_analysis,
            analysis,
            style,
            rng,
            cfg,
            notation,
        )

        reconsideration = ""
        if cfg.get("include_reconsideration", True):
            if rng.random() < cfg.get("reconsideration_prob", 0.4):
                reconsideration = self._reconsideration_line(rng)

        dead_end = ""
        if cfg.get("include_dead_end", True):
            if rng.random() < cfg.get("dead_end_prob", 0.4):
                dead_end = self._dead_end_line(candidates, best_analysis, rng, notation)

        comparison = ""
        if cfg.get("include_comparison", True):
            if rng.random() < cfg.get("comparison_prob", 0.5):
                comparison = self._comparison_line(candidates, best_analysis, rng, notation)

        conclusion = self._conclusion_line(
            board,
            best_analysis,
            analysis,
            decision_profile,
            style,
            rng,
            cfg,
            notation,
        )

        sections = [
            ("orientation", orientation),
            ("opening_plan", opening_plan_section),
            ("fen_walkthrough", fen_section),
            ("context_hint", context_section),
            ("assessment", assessment),
            ("decision_profile", decision_section),
            ("static_eval", static_eval),
            ("endgame_technique", endgame_section),
            ("threat_scan", threat_scan),
            ("plan", plan_line),
            ("opponent", opponent_line),
            ("candidates", candidates_line),
            ("avoid_blunder", avoid_section),
            ("exploration", "\n\n".join(exploration_paragraphs) if exploration_paragraphs else ""),
            ("reconsideration", reconsideration),
            ("dead_end", dead_end),
            ("comparison", comparison),
            ("conclusion", conclusion),
        ]

        text = self._apply_budget(sections, exploration_paragraphs, cfg)
        return self._apply_guardrails(text, tablebase_info, cfg)

    def _static_eval_line(
        self,
        board: chess.Board,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        max_items = int(cfg.get("max_static_eval_notes", 2))
        if max_items <= 0:
            return ""

        notes: List[str] = []
        if cfg.get("static_eval_include_material", True):
            note = self._material_note(board, rng)
            if note:
                notes.append(note)
        if cfg.get("static_eval_include_pawn_structure", True):
            note = self._pawn_structure_note(board, rng)
            if note:
                notes.append(note)
        if cfg.get("static_eval_include_king_safety", True):
            note = self._king_safety_note(board, rng)
            if note:
                notes.append(note)
        if cfg.get("static_eval_include_development", True):
            note = self._development_note(board, rng)
            if note:
                notes.append(note)
        if cfg.get("static_eval_include_mobility", True):
            note = self._mobility_note(board, rng, cfg)
            if note:
                notes.append(note)

        if not notes:
            return ""

        rng.shuffle(notes)
        chosen = notes[:max_items]
        prefix = rng.choice([
            "Position snapshot:",
            "Quick positional picture:",
            "A few static notes:",
        ])
        if style == "concise":
            prefix = rng.choice([
                "Quick note:",
                "Snapshot:",
            ])
        return f"{prefix} " + " ".join(chosen)

    def _endgame_technique_line(
        self,
        board: chess.Board,
        tablebase: Optional[TablebaseInfo],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if not bool(cfg.get("include_endgame_technique", True)):
            return ""
        if self._position_phase(board) != "endgame":
            return ""

        prob = float(cfg.get("endgame_technique_prob", 0.30))
        if style == "quick":
            prob *= 0.25
        elif style == "concise":
            prob *= 0.6
        if rng.random() >= min(max(prob, 0.0), 1.0):
            return ""

        side = "White" if board.turn == chess.WHITE else "Black"

        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        minors = (
            len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        )

        endgame_type = "endgame"
        if queens == 0 and rooks == 0 and minors == 0:
            endgame_type = "pawn endgame"
        elif queens == 0 and rooks > 0 and minors == 0:
            endgame_type = "rook endgame"
        elif queens == 0 and rooks == 0 and minors > 0:
            endgame_type = "minor-piece endgame"
        elif queens > 0 and rooks == 0:
            endgame_type = "queen endgame"
        elif queens == 0 and rooks > 0 and minors > 0:
            endgame_type = "rook + minor endgame"

        passed = self._find_passed_pawn(board, board.turn)
        opp_passed = self._find_passed_pawn(board, not board.turn)

        opposite_bishops = False
        wb = list(board.pieces(chess.BISHOP, chess.WHITE))
        bb = list(board.pieces(chess.BISHOP, chess.BLACK))
        if len(wb) == 1 and len(bb) == 1:
            wb_color = (chess.square_file(wb[0]) + chess.square_rank(wb[0])) % 2
            bb_color = (chess.square_file(bb[0]) + chess.square_rank(bb[0])) % 2
            opposite_bishops = wb_color != bb_color

        base_hints: List[str] = []
        if endgame_type == "pawn endgame":
            base_hints = [
                "King activity and tempi matter—look for opposition and passed-pawn races.",
                "Count pawn moves carefully; zugzwang and opposition often decide pawn endgames.",
                "Create a passed pawn if possible and escort it with the king.",
            ]
        elif endgame_type == "rook endgame":
            base_hints = [
                "Activate the king and keep the rook active; cutting off the enemy king is often key.",
                "Push passed pawns with the rook behind them and watch for counterplay checks.",
                "Be careful with pawn weaknesses—rook endgames swing quickly with activity.",
            ]
        elif endgame_type == "minor-piece endgame":
            base_hints = [
                "Improve king activity and target pawn weaknesses; piece activity is everything.",
                "Try to create an outside passed pawn to distract the king.",
                "Trade into a favorable pawn ending only when you’ve calculated the race.",
            ]
        elif endgame_type == "queen endgame":
            base_hints = [
                "Watch perpetual checks and keep your king safe—queen endgames are tactically sharp.",
                "Centralize the queen and look for checking nets or pawn-advance tactics.",
                "Avoid unnecessary pawn grabs if it exposes the king to perpetual checks.",
            ]
        else:
            base_hints = [
                "Activate the king and improve piece activity; endgames reward coordination.",
                "Look for a clear plan: create a passed pawn or target a structural weakness.",
                "Simplify only if it clearly improves your winning chances or defensive resources.",
            ]

        if opposite_bishops:
            base_hints.append("Opposite-colored bishops can be drawish—passed pawns and king activity matter a lot.")

        if passed is not None:
            base_hints.append(f"With a passed pawn on {chess.square_name(passed)}, look for a safe way to advance it.")
        elif opp_passed is not None:
            base_hints.append(f"Opponent has a passed pawn on {chess.square_name(opp_passed)}—stopping it may be priority #1.")

        tb_hint = ""
        if tablebase is not None:
            if tablebase.result == "winning":
                tb_hint = rng.choice(
                    [
                        f"Tablebase: {side} is winning with perfect play—focus on conversion technique.",
                        f"Tablebase says this is winning for {side}; simplify carefully and convert.",
                    ]
                )
            elif tablebase.result == "drawing":
                tb_hint = rng.choice(
                    [
                        "Tablebase: this is a draw with perfect play—avoid overpressing into a loss.",
                        "Tablebase draw—keep it solid and don’t allow unnecessary counterplay.",
                    ]
                )
            elif tablebase.result == "losing":
                tb_hint = rng.choice(
                    [
                        f"Tablebase: {side} is losing with perfect play—look for practical chances and fortress ideas.",
                        f"Tablebase says this is lost for {side}; aim for active defense and complications.",
                    ]
                )

        prefix = rng.choice(
            [
                "Endgame technique:",
                f"{endgame_type.title()} note:",
                "Technique note:",
            ]
        )
        hint = rng.choice(base_hints) if base_hints else ""
        parts = [prefix, hint]
        if tb_hint and rng.random() < float(cfg.get("endgame_tablebase_hint_prob", 0.5)):
            parts.append(tb_hint)
        return " ".join(p for p in parts if p).strip()

    def _material_note(self, board: chess.Board, rng: random.Random) -> str:
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        white = 0
        black = 0
        for piece_type, value in values.items():
            white += len(board.pieces(piece_type, chess.WHITE)) * value
            black += len(board.pieces(piece_type, chess.BLACK)) * value
        diff = white - black
        if diff == 0:
            return rng.choice([
                "Material looks even.",
                "Material is roughly equal.",
            ])
        leader = "White" if diff > 0 else "Black"
        points = abs(diff)
        if points == 1:
            return f"{leader} is up a pawn."
        if points == 2:
            return f"{leader} has a small material edge."
        return f"{leader} is up material."

    def _pawn_structure_note(self, board: chess.Board, rng: random.Random) -> str:
        color = board.turn
        opp = not color
        weak = self._pawn_weakness_summary(board, color)
        opp_weak = self._pawn_weakness_summary(board, opp)
        islands = self._pawn_islands(board, color)
        opp_islands = self._pawn_islands(board, opp)

        if weak:
            return weak
        if islands > opp_islands:
            return rng.choice([
                "Pawn structure is a bit fragmented.",
                "The pawn structure looks slightly split into islands.",
            ])
        if islands < opp_islands:
            return rng.choice([
                "Pawn structure looks cohesive.",
                "The pawn structure feels well-connected.",
            ])
        if opp_weak:
            return rng.choice([
                "The opponent's pawn structure has some targets.",
                "There may be pawn weaknesses to play against.",
            ])
        return rng.choice([
            "No obvious pawn weaknesses jump out.",
            "Pawn structure looks fairly healthy.",
        ])

    def _pawn_weakness_summary(self, board: chess.Board, color: bool) -> str:
        isolated = self._find_isolated_pawn(board, color)
        if isolated is not None:
            square = chess.square_name(isolated)
            return f"An isolated pawn on {square} could be a long-term weakness."
        doubled = self._find_doubled_pawn_file(board, color)
        if doubled is not None:
            file_letter = "abcdefgh"[doubled]
            return f"Doubled pawns on the {file_letter}-file may become targets."
        backward = self._find_backward_pawn(board, color)
        if backward is not None:
            square = chess.square_name(backward)
            return f"A backward pawn on {square} can be hard to defend."
        return ""

    def _pawn_islands(self, board: chess.Board, color: bool) -> int:
        pawn_files = [False] * 8
        for sq in board.pieces(chess.PAWN, color):
            pawn_files[chess.square_file(sq)] = True
        islands = 0
        in_island = False
        for present in pawn_files:
            if present and not in_island:
                islands += 1
                in_island = True
            elif not present:
                in_island = False
        return islands

    def _king_safety_note(self, board: chess.Board, rng: random.Random) -> str:
        side = "White" if board.turn == chess.WHITE else "Black"
        opp_side = "Black" if board.turn == chess.WHITE else "White"
        if self._is_exposed_king(board, board.turn):
            return rng.choice([
                f"{side}'s king safety looks a bit loose.",
                f"{side}'s king is somewhat exposed.",
            ])
        if self._is_exposed_king(board, not board.turn):
            return rng.choice([
                f"{opp_side}'s king safety may be a concern.",
                f"{opp_side}'s king looks a bit drafty.",
            ])
        return rng.choice([
            "Both kings look reasonably safe for now.",
            "King safety looks stable on both sides.",
        ])

    def _development_note(self, board: chess.Board, rng: random.Random) -> str:
        if self._position_phase(board) != "opening":
            return ""
        color = board.turn
        undeveloped = self._count_undeveloped_minors(board, color)
        if undeveloped >= 3:
            return rng.choice([
                "Several minor pieces still need development.",
                "Development is still getting started; pieces need to come out.",
            ])
        if undeveloped <= 1:
            return rng.choice([
                "Development is largely complete.",
                "Most pieces are developed already.",
            ])
        return rng.choice([
            "There is still some development to finish.",
            "A couple of pieces could still improve.",
        ])

    def _count_undeveloped_minors(self, board: chess.Board, color: bool) -> int:
        if color == chess.WHITE:
            start_squares = [chess.B1, chess.G1, chess.C1, chess.F1]
        else:
            start_squares = [chess.B8, chess.G8, chess.C8, chess.F8]
        count = 0
        for sq in start_squares:
            piece = board.piece_at(sq)
            if not piece or piece.color != color:
                continue
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                count += 1
        return count

    def _mobility_note(self, board: chess.Board, rng: random.Random, cfg: Dict[str, Any]) -> str:
        if board.is_check():
            return ""
        mobility_threshold = int(cfg.get("mobility_note_threshold", 8))
        my_moves = sum(1 for _ in board.legal_moves)
        opp_board = board.copy()
        opp_board.turn = not board.turn
        opp_moves = sum(1 for _ in opp_board.legal_moves)
        delta = my_moves - opp_moves
        if delta >= mobility_threshold:
            return rng.choice([
                "Side to move has more options and activity.",
                "There is a bit more space to maneuver here.",
            ])
        if delta <= -mobility_threshold:
            return rng.choice([
                "Side to move looks a bit cramped with fewer options.",
                "Options are limited, so precision matters.",
            ])
        return rng.choice([
            "Mobility looks fairly balanced.",
            "Both sides have a similar number of options.",
        ])

    def _resolve_config(self, source: Optional[str]) -> Dict[str, Any]:
        cfg = dict(self.config)
        source_overrides = self.config.get("source_overrides", {})
        if source and source in source_overrides:
            cfg.update(source_overrides[source] or {})
        return cfg

    def _select_candidates(
        self,
        analysis: Any,
        cfg: Dict[str, Any],
        rng: random.Random,
    ) -> List[Any]:
        if not analysis.move_analyses:
            return []
        min_c = max(1, int(cfg.get("min_candidates", 3)))
        max_c = max(min_c, int(cfg.get("max_candidates", 5)))
        max_c = min(max_c, len(analysis.move_analyses))
        count = rng.randint(min_c, max_c) if max_c > min_c else max_c

        pool_size = cfg.get("candidate_pool_size")
        pool_cap = max_c
        if pool_size is not None:
            try:
                pool_cap = int(pool_size)
            except (TypeError, ValueError):
                pool_cap = max_c
        pool_cap = max(count, min(pool_cap, len(analysis.move_analyses)))
        pool = list(analysis.move_analyses[:pool_cap])

        sampling = str(cfg.get("candidate_pool_sampling", "top")).lower()
        if sampling == "top" or pool_cap <= count:
            return pool[:count]
        if sampling == "uniform":
            return rng.sample(pool, count)
        if sampling == "softmax":
            temperature = float(cfg.get("candidate_pool_temperature", 1.0))
            temperature = max(1e-6, temperature)
            weights = []
            for cand in pool:
                wp = getattr(cand, "win_probability", None)
                if wp is None:
                    weights.append(1.0)
                else:
                    weights.append(math.exp(float(wp) / temperature))
            return self._weighted_sample_without_replacement(pool, weights, count, rng)

        return rng.sample(pool, count)

    @staticmethod
    def _weighted_sample_without_replacement(
        items: List[Any],
        weights: List[float],
        k: int,
        rng: random.Random,
    ) -> List[Any]:
        """Sample k unique items, roughly proportional to weights."""
        if k <= 0:
            return []
        chosen = []
        remaining_items = list(items)
        remaining_weights = [max(0.0, float(w)) for w in weights]
        k = min(k, len(remaining_items))

        for _ in range(k):
            total = sum(remaining_weights)
            if total <= 0:
                idx = rng.randrange(len(remaining_items))
            else:
                pick = rng.random() * total
                running = 0.0
                idx = 0
                for i, w in enumerate(remaining_weights):
                    running += w
                    if pick <= running:
                        idx = i
                        break
            chosen.append(remaining_items.pop(idx))
            remaining_weights.pop(idx)

        return chosen

    def _orientation_line(
        self,
        board: chess.Board,
        opening: Optional[OpeningInfo],
        tablebase: Optional[TablebaseInfo],
        source: Optional[str],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        side = "White" if board.turn == chess.WHITE else "Black"
        phase = self._position_phase(board)
        opener = ""
        if opening:
            opener = f"Opening: {opening.eco} {opening.name}."
        elif phase == "endgame":
            opener = "This is an endgame position."
        elif phase == "opening":
            opener = "Opening phase of the game."

        source_hint = ""
        if source == "puzzle":
            source_hint = rng.choice([
                "This looks like a tactical puzzle.",
                "Puzzle position with a likely forcing line.",
                "Tactical spot where precision matters.",
            ])
        elif source == "game":
            source_hint = rng.choice([
                "Game position with practical choices.",
                "From a game, so steady development matters.",
                "Practical game position.",
            ])

        lines = []
        starter_prob = float(cfg.get("orientation_starter_prob", 0.25))
        if rng.random() < min(max(starter_prob, 0.0), 1.0):
            lines.append(rng.choice(PHRASE_LIBRARY["orientation_starters"]))
        if opener:
            lines.append(opener)
        if source_hint:
            lines.append(source_hint)
        lines.append(f"{side} to move.")
        return " ".join(lines)

    def _opening_plan_line(
        self,
        opening: Optional[OpeningInfo],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if not opening or not bool(cfg.get("include_opening_plan_hints", True)):
            return ""

        prob = float(cfg.get("opening_plan_prob", 0.35))
        if style == "quick":
            prob *= 0.25
        elif style == "concise":
            prob *= 0.6
        if rng.random() >= min(max(prob, 0.0), 1.0):
            return ""

        name = (opening.name or "").strip()
        name_lc = name.lower()
        short = name.split(":")[0].split(",")[0].strip() if name else opening.eco
        short = short or opening.eco or "opening"

        hints: List[str] = []
        if "sicilian" in name_lc:
            hints = [
                "Contest the center (especially d4) and develop quickly; tactics can appear early.",
                "Keep an eye on development and king safety—Sicilian positions can get sharp fast.",
                "Focus on piece activity and typical central breaks; avoid falling behind in development.",
            ]
        elif "french" in name_lc:
            hints = [
                "Play around the central pawn chain and watch the light-squared bishop.",
                "Undermine the center with timely pawn breaks and keep pieces coordinated.",
                "Space vs counterplay: be patient and look for the right pawn break timing.",
            ]
        elif "caro-kann" in name_lc:
            hints = [
                "Aim for a solid structure, develop smoothly, and look for the right central break.",
                "Stay solid and active—don’t get passive behind the pawn chain.",
                "Prioritize piece development and a clean endgame structure.",
            ]
        elif "pirc" in name_lc or "modern" in name_lc:
            hints = [
                "Let the center form, then undermine it with well-timed pawn breaks.",
                "Development and king safety first; counterplay comes from striking the center.",
                "Be ready to challenge the center and avoid getting cramped.",
            ]
        elif "king's indian" in name_lc or "kings indian" in name_lc:
            hints = [
                "Expect a central confrontation—prepare the right pawn break and stay coordinated.",
                "Keep an eye on kingside initiative and central tension.",
                "Space matters; look for the moment to strike at the center.",
            ]
        elif "grünfeld" in name_lc or "grunfeld" in name_lc:
            hints = [
                "Pressure the center and target d4; active piece play is key.",
                "Be precise: the center can be strong but also a target.",
                "Look for dynamic counterplay rather than passive defense.",
            ]
        elif "slav" in name_lc:
            hints = [
                "Keep the structure solid and focus on harmonious development.",
                "Develop the light-squared bishop thoughtfully and contest the center.",
                "Play for steady improvement and good piece coordination.",
            ]
        elif "queen's gambit" in name_lc or "queens gambit" in name_lc:
            hints = [
                "Fight for central control and look for long-term structural targets.",
                "Focus on development and the central break timing.",
                "Play for small, stable advantages—structure and activity both matter.",
            ]
        elif "catalan" in name_lc:
            hints = [
                "Use the long diagonal pressure and aim for stable positional pressure.",
                "Prioritize development and long-term pressure on queenside squares.",
                "Keep the center flexible and maintain piece activity.",
            ]
        elif "nimzo" in name_lc or "nimzo-indian" in name_lc:
            hints = [
                "Central tension and structure matter—aim for active pieces and clear plans.",
                "Watch pawn structure decisions; they often define the middlegame plan.",
                "Play for pressure and development rather than grabbing material.",
            ]
        elif "ruy lopez" in name_lc or "spanish" in name_lc:
            hints = [
                "Build slowly: improve pieces, keep the center stable, and look for the right break.",
                "Plan around pressure on e5 and long-term piece placement.",
                "Keep flexibility—Spanish positions reward patience and coordination.",
            ]
        elif "italian" in name_lc:
            hints = [
                "Develop quickly and watch tactical ideas around f7 and the center.",
                "Prioritize king safety and central control; tactics can appear quickly.",
                "Look for active piece placement and direct threats if the opponent lags in development.",
            ]
        elif "scotch" in name_lc:
            hints = [
                "Open the center early and develop with tempo.",
                "Central clarity matters—use the open lines for active pieces.",
                "Avoid slow moves; development speed is a big factor here.",
            ]
        elif "english" in name_lc:
            hints = [
                "Stay flexible and watch transpositions; build pressure step by step.",
                "Clamp key squares and aim for a favorable pawn break.",
                "Prioritize piece activity and long-term structure over immediate tactics.",
            ]
        elif "london" in name_lc:
            hints = [
                "Stick to the setup, develop smoothly, and look for a clean kingside plan.",
                "Don’t rush—improve pieces and avoid creating unnecessary weaknesses.",
                "Focus on a clear plan and steady development rather than early tactics.",
            ]
        elif "dutch" in name_lc:
            hints = [
                "Watch king safety and dark-square weaknesses; coordination is important.",
                "Build kingside initiative carefully and avoid overextending.",
                "Make sure development keeps up with pawn advances.",
            ]
        else:
            hints = [
                "Develop smoothly, keep the king safe, and don’t concede the center for free.",
                "Prioritize development and coordination; avoid early pawn grabs.",
                "Keep pieces active and look for a clean plan into the middlegame.",
            ]

        prefix = rng.choice(
            [
                "Opening idea:",
                "Opening plan:",
                f"Opening plan ({short}):",
                f"Typical idea ({short}):",
            ]
        )
        return f"{prefix} {rng.choice(hints)}"

    def _fen_walkthrough_line(
        self,
        board: chess.Board,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        """
        Generate a short "what the FEN says" walkthrough.

        This is meant to add occasional structured variety by explicitly calling
        out the metadata encoded in FEN (turn, castling rights, en passant, etc.)
        plus a small amount of piece-placement summarization.
        """
        max_items = int(cfg.get("fen_walkthrough_max_items", 4))
        if max_items <= 0:
            return ""

        if style == "quick":
            max_items = min(max_items, 2)
        elif style == "concise":
            max_items = min(max_items, 3)

        side = "White" if board.turn == chess.WHITE else "Black"
        move_no = int(getattr(board, "fullmove_number", 0) or 0)

        items: List[str] = []

        if cfg.get("fen_walkthrough_include_turn", True):
            if move_no > 0:
                items.append(
                    rng.choice(
                        [
                            f"{side} to play on move {move_no}.",
                            f"Move {move_no}, {side} to play.",
                            f"It's {side} to move (move {move_no}).",
                        ]
                    )
                )
            else:
                items.append(rng.choice([f"{side} to move.", f"{side} to play."]))

        if cfg.get("fen_walkthrough_include_check", True) and board.is_check():
            items.append(rng.choice(["Side to move is in check.", f"{side} is in check and must respond."]))

        if cfg.get("fen_walkthrough_include_castling", True):
            white_castle = self._castling_rights(board, chess.WHITE)
            black_castle = self._castling_rights(board, chess.BLACK)
            items.append(
                rng.choice(
                    [
                        f"Castling rights: White {white_castle}, Black {black_castle}.",
                        f"Castling: White {white_castle}; Black {black_castle}.",
                    ]
                )
            )

        if cfg.get("fen_walkthrough_include_en_passant", True):
            if board.ep_square is None:
                items.append(rng.choice(["En passant target: none.", "No en passant square is available."]))
            else:
                ep = chess.square_name(board.ep_square)
                items.append(rng.choice([f"En passant target square is {ep}.", f"En passant is possible on {ep}."]))

        if cfg.get("fen_walkthrough_include_kings", True):
            wk = board.king(chess.WHITE)
            bk = board.king(chess.BLACK)
            if wk is not None and bk is not None:
                wk_name = chess.square_name(wk)
                bk_name = chess.square_name(bk)
                items.append(rng.choice([f"Kings: White {wk_name}, Black {bk_name}.", f"King locations: {wk_name} vs {bk_name}."]))

        if cfg.get("fen_walkthrough_include_queens", True):
            wq = [chess.square_name(sq) for sq in board.pieces(chess.QUEEN, chess.WHITE)]
            bq = [chess.square_name(sq) for sq in board.pieces(chess.QUEEN, chess.BLACK)]
            if not wq and not bq:
                items.append(rng.choice(["Queens are off the board.", "No queens remain."]))
            elif wq and not bq:
                items.append(f"Only White has a queen ({', '.join(wq)}).")
            elif bq and not wq:
                items.append(f"Only Black has a queen ({', '.join(bq)}).")
            else:
                items.append(f"Queens are on ({', '.join(wq)} vs {', '.join(bq)}).")

        if cfg.get("fen_walkthrough_include_material", False):
            items.append(self._material_note(board, rng))

        if cfg.get("fen_walkthrough_include_piece_count", False):
            items.append(f"Pieces on board: {len(board.piece_map())}.")

        items = [item for item in items if item]
        if not items:
            return ""

        rng.shuffle(items)
        count = min(max_items, len(items))
        if count >= 3:
            count = rng.randint(2, count)
        chosen = items[:count]

        intro = rng.choice(
            [
                "Reading the FEN:",
                "FEN quick read:",
                "From the FEN:",
                "FEN decode:",
                "What the FEN encodes:",
            ]
        )
        format_mode = str(cfg.get("fen_walkthrough_format", "auto")).lower()
        if format_mode not in ("auto", "bullets", "sentences"):
            format_mode = "auto"

        if format_mode == "auto":
            if style in ("quick", "concise"):
                format_mode = "sentences"
            else:
                format_mode = "bullets" if rng.random() < 0.55 else "sentences"

        if format_mode == "bullets":
            return f"{intro}\n- " + "\n- ".join(chosen)
        return f"{intro} " + " ".join(chosen)

    @staticmethod
    def _castling_rights(board: chess.Board, color: bool) -> str:
        rights: List[str] = []
        if board.has_kingside_castling_rights(color):
            rights.append("O-O")
        if board.has_queenside_castling_rights(color):
            rights.append("O-O-O")
        return " and ".join(rights) if rights else "none"

    def _assessment_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        tablebase: Optional[TablebaseInfo],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if tablebase:
            side = "White" if board.turn == chess.WHITE else "Black"
            if tablebase.result == "drawing":
                return rng.choice([
                    "With perfect play, this is a draw.",
                    "Tablebase says the position is drawn.",
                ])
            if tablebase.result == "winning":
                return rng.choice([
                    f"With perfect play, {side} is winning.",
                    f"Tablebase indicates a win for {side}.",
                ])
            if tablebase.result == "losing":
                return rng.choice([
                    f"With perfect play, {side} is losing.",
                    f"Tablebase indicates {side} is losing.",
                ])
            return "Tablebase result is unclear."
        if best_analysis is None:
            return "No clear evaluation, so focus on solid moves."
        win_prob = getattr(best_analysis, "win_probability", 0.5)
        unclear_margin = float(cfg.get("unclear_win_margin", 0.04))
        if abs(win_prob - 0.5) <= unclear_margin:
            base = rng.choice([
                "The position looks balanced and unclear.",
                "This feels roughly equal without a clear edge.",
                "Neither side has much of an edge here.",
            ])
            return self._append_positional_cues(base, board, cfg)
        if win_prob >= 0.60:
            base = rng.choice([
                "Side to move has the initiative and can press.",
                "There is a clear edge for the side to move.",
            ])
            return self._append_positional_cues(base, board, cfg)
        if win_prob <= 0.40:
            base = rng.choice([
                "The side to move is under pressure here.",
                "It looks a bit worse, so accuracy matters.",
            ])
            return self._append_positional_cues(base, board, cfg)
        base = rng.choice([
            "The position looks roughly balanced.",
            "This feels quite equal with chances for both sides.",
        ])
        return self._append_positional_cues(base, board, cfg)

    def _candidate_line(
        self,
        candidates: List[Any],
        style: str,
        rng: random.Random,
        notation: str,
    ) -> str:
        if not candidates:
            return "Candidates are unclear in this position."
        moves = [self._move_label(cand, notation) for cand in candidates]
        prefix = rng.choice(PHRASE_LIBRARY["candidate_intros"])
        return f"{prefix} {', '.join(moves)}."

    def _avoid_move_line(
        self,
        board: chess.Board,
        analysis: Any,
        best_analysis: Optional[Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """
        Emit an occasional "what to avoid" warning.

        Heuristic: find a move that is relatively likely under the move-prob
        distribution but has large centipawn loss vs the best move, and (if PV
        is available) cite the opponent's first refutation move.

        If a refutation PV is available, also add a short explanation of what
        goes wrong (e.g., drops a queen, gets mated, runs into a fork).
        """
        if not bool(cfg.get("include_avoid_blunders", True)):
            return ""

        prob = float(cfg.get("avoid_blunder_prob", 0.25))
        if style == "quick":
            prob *= 0.25
        elif style == "concise":
            prob *= 0.6
        if rng.random() >= min(max(prob, 0.0), 1.0):
            return ""

        min_cp_loss = int(cfg.get("avoid_min_cp_loss", 120))
        min_move_prob = float(cfg.get("avoid_min_move_prob", 0.05))

        move_probs: Dict[str, float] = dict(getattr(analysis, "move_probs", None) or {})
        best_uci = getattr(best_analysis, "uci", None) if best_analysis is not None else None

        shallow_cps = getattr(analysis, "shallow_move_cps", None) or {}
        best_shallow_cp = None
        if shallow_cps:
            values = [v for v in shallow_cps.values() if v is not None]
            best_shallow_cp = max(values) if values else None

        chosen: Optional[Any] = None
        chosen_score: Tuple[int, float, int] = (-1, 0.0, -1)
        for cand in getattr(analysis, "move_analyses", None) or []:
            if best_uci and getattr(cand, "uci", None) == best_uci:
                continue
            try:
                cp_loss = int(getattr(cand, "cp_loss", 0))
            except (TypeError, ValueError):
                continue
            if cp_loss < min_cp_loss:
                continue
            p = float(move_probs.get(getattr(cand, "uci", ""), 0.0) or 0.0)
            if p < min_move_prob:
                continue
            tempting_score = 0
            try:
                mv = chess.Move.from_uci(getattr(cand, "uci", ""))
                if mv in board.legal_moves:
                    if board.gives_check(mv):
                        tempting_score += 3
                    if board.is_capture(mv):
                        tempting_score += 2
                    if mv.promotion:
                        tempting_score += 3
                    # Quiet "hits something" moves often look attractive to humans.
                    if not (board.is_capture(mv) or board.gives_check(mv) or mv.promotion or board.is_castling(mv)):
                        features = self._move_features(board, mv, cfg)
                        if any("targets the" in f for f in features):
                            tempting_score += 1
                # Shallow-vs-deep disagreements are the archetypal "looks good, fails" blunders.
                shallow_cp = shallow_cps.get(getattr(cand, "uci", ""))
                deep_cp = getattr(cand, "centipawn", None)
                if best_shallow_cp is not None and shallow_cp is not None and deep_cp is not None:
                    shallow_good = shallow_cp >= best_shallow_cp - int(cfg.get("trap_shallow_good_cp", 30))
                    deep_bad = deep_cp <= best_shallow_cp - int(cfg.get("trap_shallow_bad_cp", 120))
                    if shallow_good and deep_bad:
                        tempting_score += 3
            except Exception:
                pass

            score = (tempting_score, p, cp_loss)
            if chosen is None or score > chosen_score:
                chosen = cand
                chosen_score = score

        if chosen is None:
            return ""

        blunder_label = self._move_label(chosen, notation)

        temptation_reason = ""
        bad_result = ""
        refutation_line = ""

        reply_label = ""
        pv_uci = list(getattr(chosen, "pv_uci", None) or [])
        start_board = board
        if pv_uci and pv_uci[0] == getattr(chosen, "uci", ""):
            pv_uci = pv_uci[1:]
            try:
                start_board = board.copy()
                start_board.push(chess.Move.from_uci(getattr(chosen, "uci", "")))
            except Exception:
                start_board = board
        if pv_uci:
            reply_uci = pv_uci[0]
            try:
                reply_move = chess.Move.from_uci(reply_uci)
                reply_label = self._move_notation(start_board, reply_move, notation)
            except Exception:
                reply_label = reply_uci

        try:
            blunder_move = chess.Move.from_uci(getattr(chosen, "uci", ""))
            if blunder_move in board.legal_moves:
                temptation_reason = self._tempting_blunder_reason(board, blunder_move, chosen, analysis, rng, cfg)
        except Exception:
            temptation_reason = ""

        try:
            if reply_label and pv_uci:
                bad_result = self._blunder_bad_result(
                    board=board,
                    blunder_cand=chosen,
                    start_board_after_blunder=start_board,
                    reply_uci=pv_uci[0],
                    rng=rng,
                    cfg=cfg,
                    notation=notation,
                )
        except Exception:
            bad_result = ""

        if reply_label:
            refutation_line = self._trap_refutation_line(board, chosen, notation, cfg)
            if style in ("quick", "concise") and refutation_line:
                # Keep short styles short.
                refutation_line = ""

            if temptation_reason and bad_result:
                template = rng.choice(
                    [
                        "Avoid {move}; it looks tempting because it {why_good}, but after {reply} it {why_bad}.",
                        "Be careful with {move}: it {why_good}, but {reply} punishes it and it {why_bad}.",
                        "Try not to play {move}; it {why_good}, but after {reply} it {why_bad}.",
                    ]
                )
                base = template.format(
                    move=blunder_label,
                    why_good=temptation_reason,
                    reply=reply_label,
                    why_bad=bad_result,
                )
                return " ".join(p for p in (base, refutation_line) if p).strip()

            template = rng.choice(
                [
                    "Avoid {move}; it allows {reply}.",
                    "Be careful with {move}: {reply} is the refutation.",
                    "Try not to play {move}; after that, {reply} is strong.",
                    "Don't go for {move}; it runs into {reply}.",
                ]
            )
            return template.format(move=blunder_label, reply=reply_label)

        template = rng.choice(
            [
                "Avoid {move}; it gives up too much.",
                "Be careful: {move} is a serious inaccuracy here.",
                "Try not to play {move}; it loses a lot of ground.",
            ]
        )
        return template.format(move=blunder_label)

    def _tempting_blunder_reason(
        self,
        board: chess.Board,
        move: chess.Move,
        cand: Any,
        analysis: Any,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        """
        Describe why a blunder might look appealing to a human.

        Keep this as a short verb phrase that follows "it ...".
        """
        shallow_cps = getattr(analysis, "shallow_move_cps", None) or {}
        if shallow_cps and getattr(cand, "uci", "") in shallow_cps:
            values = [v for v in shallow_cps.values() if v is not None]
            best_shallow = max(values) if values else None
            shallow = shallow_cps.get(getattr(cand, "uci", ""))
            deep = getattr(cand, "centipawn", None)
            if best_shallow is not None and shallow is not None and deep is not None:
                shallow_good = shallow >= best_shallow - int(cfg.get("trap_shallow_good_cp", 30))
                deep_bad = deep <= best_shallow - int(cfg.get("trap_shallow_bad_cp", 120))
                if shallow_good and deep_bad and rng.random() < 0.6:
                    return rng.choice(
                        [
                            "looks fine at shallow depth",
                            "passes a shallow check",
                            "looks good in a quick scan",
                        ]
                    )

        if board.gives_check(move):
            return rng.choice(["gives check", "creates a forcing check", "looks forcing with check"])
        if board.is_capture(move):
            capture_square, captured = self._captured_piece(board, move)
            if captured is not None:
                name = PIECE_NAMES.get(captured.piece_type, "piece")
                sq = chess.square_name(capture_square)
                return rng.choice(
                    [
                        f"wins the {name} on {sq}",
                        f"grabs the {name} on {sq}",
                        f"snaps off the {name} on {sq}",
                    ]
                )
            return rng.choice(["wins material", "grabs material", "looks like a free capture"])
        if move.promotion:
            if move.promotion == chess.QUEEN:
                return rng.choice(["promotes immediately", "queens a pawn", "forces promotion"])
            promo = PIECE_NAMES.get(move.promotion, "piece")
            return rng.choice([f"underpromotes to a {promo}", "promotes tactically"])

        features = self._move_features(board, move, cfg)
        if features:
            feature = features[0]
            return rng.choice(
                [
                    feature,
                    f"{feature} and looks natural",
                ]
            )
        return rng.choice(["looks natural", "feels active", "seems to improve the position"])

    def _blunder_bad_result(
        self,
        *,
        board: chess.Board,
        blunder_cand: Any,
        start_board_after_blunder: chess.Board,
        reply_uci: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """
        Describe what goes wrong after the opponent's PV reply.

        Keep this as a short verb phrase that follows "it ...".
        """
        mate_in = getattr(blunder_cand, "mate_in", None)
        if isinstance(mate_in, int) and mate_in < 0:
            return f"allows mate in {abs(mate_in)}"

        try:
            reply_move = chess.Move.from_uci(reply_uci)
        except Exception:
            return rng.choice(["gets punished tactically", "loses material", "collapses"])

        if reply_move not in start_board_after_blunder.legal_moves:
            return rng.choice(["gets punished tactically", "loses material", "collapses"])

        after_reply = start_board_after_blunder.copy(stack=False)
        gives_check = start_board_after_blunder.gives_check(reply_move)
        is_capture = start_board_after_blunder.is_capture(reply_move)
        capture_square, captured = self._captured_piece(start_board_after_blunder, reply_move) if is_capture else (None, None)
        after_reply.push(reply_move)

        if after_reply.is_checkmate():
            return rng.choice(["gets checkmated", "is checkmate"])

        if captured is not None and captured.color == board.turn:
            name = PIECE_NAMES.get(captured.piece_type, "piece")
            sq = chess.square_name(capture_square) if capture_square is not None else None
            if captured.piece_type == chess.QUEEN:
                return rng.choice(
                    [
                        "drops the queen",
                        "loses the queen",
                        f"loses the queen on {sq}" if sq else "loses the queen",
                    ]
                )
            if captured.piece_type == chess.ROOK:
                return rng.choice(
                    [
                        "drops a rook",
                        f"loses the rook on {sq}" if sq else "loses a rook",
                        "loses a full exchange",
                    ]
                )
            if captured.piece_type in (chess.BISHOP, chess.KNIGHT):
                return rng.choice(
                    [
                        "drops a piece",
                        f"loses the {name} on {sq}" if sq else f"loses a {name}",
                        "loses a minor piece",
                    ]
                )
            return rng.choice(
                [
                    f"loses the {name}" if not sq else f"loses the {name} on {sq}",
                    "bleeds material",
                ]
            )

        if gives_check:
            return rng.choice(["walks into a forcing check", "gets hit by a check and loses time"])

        hanging = self._find_hanging_pieces_for_color(after_reply, board.turn, 1)
        if hanging:
            piece, square = hanging[0]
            name = PIECE_NAMES.get(piece.piece_type, "piece")
            return rng.choice(
                [
                    f"leaves the {name} on {chess.square_name(square)} hanging",
                    f"drops the {name} on {chess.square_name(square)}",
                ]
            )

        # Fall back to a compact tactical label if we can spot one.
        try:
            fake = type("ReplyMove", (), {})()
            fake.uci = reply_uci
            fake.pv_uci = [reply_uci]
            motifs = self._detect_candidate_motifs(start_board_after_blunder, reply_move, fake, cfg, rng=rng)
            tactical = next(
                (
                    m
                    for m in motifs
                    if any(
                        key in m.lower()
                        for key in (
                            "mate",
                            "fork",
                            "pin",
                            "skewer",
                            "deflection",
                            "x-ray",
                            "xray",
                            "discovered",
                            "double",
                            "interference",
                            "back rank",
                            "trapped",
                            "overload",
                            "decoy",
                        )
                    )
                ),
                None,
            )
            if tactical:
                return f"runs into {tactical}"
        except Exception:
            pass

        return rng.choice(["gets refuted tactically", "loses material", "falls apart"])

    @staticmethod
    def _prefix_lines(prefix: str, text: str) -> str:
        if not text:
            return ""
        lines = str(text).splitlines()
        out_lines = []
        for line in lines:
            if not line.strip():
                out_lines.append("")
            else:
                out_lines.append(f"{prefix}{line}")
        return "\n".join(out_lines)

    def _dialogue_speakers(self, rng: random.Random, cfg: Dict[str, Any]) -> Tuple[str, str]:
        pairs = cfg.get("dialogue_speaker_pairs")
        if isinstance(pairs, list) and pairs:
            normalized = []
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                a, b = str(pair[0]).strip(), str(pair[1]).strip()
                if a and b:
                    normalized.append((a, b))
            if normalized:
                return rng.choice(normalized)

        return rng.choice(
            [
                ("Coach", "Student"),
                ("Analyst", "Skeptic"),
                ("Tutor", "Learner"),
            ]
        )

    def _dialogue_candidate_explanation(
        self,
        board: chess.Board,
        cand: Any,
        best_win: float,
        analysis: Any,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        move_label = self._move_label(cand, notation)
        try:
            move = chess.Move.from_uci(cand.uci)
        except Exception:
            move = None
        features = self._move_features(board, move, cfg) if move is not None else []
        idea = self._idea_phrase(features, rng, style)
        unclear_margin = float(cfg.get("unclear_win_margin", 0.04))
        quality = self._quality_phrase(cand, best_win, rng, style, unclear_margin)

        motif_line = ""
        if cfg.get("include_motifs", False) and move is not None and rng.random() < float(cfg.get("dialogue_motif_prob", 0.35)):
            motifs = self._detect_candidate_motifs(board, move, cand, cfg, rng=rng)
            if motifs:
                motif_line = self._motif_phrase(motifs[:1], rng)

        trap_line = ""
        if cfg.get("include_trap_detection", False) and rng.random() < float(cfg.get("dialogue_trap_prob", 0.40)):
            trap = self._detect_trap_for_candidate(cand, analysis, cfg)
            trap_line = self._trap_phrase(trap, rng, board, cand, notation, cfg)

        pv_line = ""
        if cfg.get("include_candidate_pv", True) and move is not None and rng.random() < float(cfg.get("dialogue_candidate_pv_prob", 0.35)):
            pv_line = self._candidate_pv_snippet(board, cand, notation, cfg, rng)

        parts = [
            f"{move_label}:",
            idea,
            quality,
            pv_line,
            motif_line,
            trap_line,
        ]
        return " ".join(p for p in parts if p).strip()

    def _generate_dialogue_style(
        self,
        *,
        board: chess.Board,
        position: Dict[str, Any],
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        decision_profile: Dict[str, Any],
        opening_info: Optional[OpeningInfo],
        tablebase_info: Optional[TablebaseInfo],
        opening_plan: str,
        fen_walkthrough: str,
        context_hint: str,
        decision_line: str,
        endgame_technique: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """
        Coherent Q/A style trace ("dialogue").

        This stays a single `reasoning_trace` string (no separate message column),
        but formats the reasoning as a short dialogue.
        """
        coach, student = self._dialogue_speakers(rng, cfg)
        coach_p = f"{coach}: "
        student_p = f"{student}: "

        phase = self._position_phase(board)
        side = "White" if board.turn == chess.WHITE else "Black"
        source = str(position.get("source") or "")
        opener_bits: List[str] = []
        if opening_info:
            opener_bits.append(f"Opening: {opening_info.eco} {opening_info.name}.")
        else:
            opener_bits.append(f"{phase.title()} position.")
        if source == "puzzle":
            opener_bits.append("Puzzle position.")
        elif source == "game":
            opener_bits.append("From a game.")
        opener_bits.append(f"{side} to move.")

        lines: List[str] = []
        lines.append(coach_p + " ".join(opener_bits))

        if opening_plan:
            lines.append(student_p + "Any opening plan I should keep in mind?")
            lines.append(coach_p + opening_plan)

        if fen_walkthrough:
            lines.append(student_p + "Can you decode the FEN quickly?")
            lines.append(self._prefix_lines(coach_p, fen_walkthrough))

        if context_hint:
            lines.append(student_p + "What should I look at first?")
            lines.append(coach_p + context_hint)

        if cfg.get("include_assessment", True):
            assessment = self._assessment_line(board, best_analysis, tablebase_info, "concise", rng, cfg)
            if assessment:
                lines.append(student_p + "What's the evaluation?")
                lines.append(coach_p + assessment)

        if decision_line:
            lines.append(student_p + "Is this a critical moment?")
            lines.append(coach_p + decision_line)

        if endgame_technique:
            lines.append(student_p + "Any endgame technique to remember here?")
            lines.append(coach_p + endgame_technique)

        threat_scan = ""
        if cfg.get("include_threat_scan", True) and rng.random() < float(cfg.get("dialogue_threat_scan_prob", 0.35)):
            threat_scan = self._threat_scan_line(board, "concise", rng, notation, cfg)
        if threat_scan:
            lines.append(student_p + "Any immediate tactics or threats?")
            lines.append(coach_p + threat_scan)

        lines.append(student_p + "What moves should we consider?")
        lines.append(coach_p + self._candidate_line(candidates, "concise", rng, notation))

        if candidates:
            max_cands = int(cfg.get("dialogue_max_candidates", 3))
            max_cands = max(1, max_cands)
            best_win = getattr(best_analysis, "win_probability", 0.5) if best_analysis else 0.5
            for cand in candidates[:max_cands]:
                move_label = self._move_label(cand, notation)
                q = rng.choice(
                    [
                        f"Does {move_label} work?",
                        f"What about {move_label}?",
                        f"How would you explain {move_label}?",
                    ]
                )
                lines.append(student_p + q)
                lines.append(coach_p + self._dialogue_candidate_explanation(board, cand, best_win, analysis, "concise", rng, cfg, notation))

        avoid_line = self._avoid_move_line(board, analysis, best_analysis, "concise", rng, cfg, notation)
        if avoid_line:
            lines.append(student_p + "Anything I should avoid?")
            lines.append(coach_p + avoid_line)

        lines.append(student_p + "So what's the best move?")
        conclusion = self._conclusion_line(
            board,
            best_analysis,
            analysis,
            decision_profile,
            "dialogue",
            rng,
            cfg,
            notation,
        )
        lines.append(self._prefix_lines(coach_p, conclusion))

        return "\n".join(lines)

    def _exploration_paragraphs(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> List[str]:
        if not candidates:
            return []
        ordered = candidates.copy()
        if cfg.get("candidate_order", "shuffled") == "shuffled":
            rng.shuffle(ordered)
        best_win = getattr(best_analysis, "win_probability", 0.5) if best_analysis else 0.5
        best_uci = getattr(best_analysis, "uci", None) if best_analysis is not None else None

        pv_budget = int(cfg.get("candidate_pv_max_per_trace", len(ordered)))
        pv_budget = max(0, pv_budget)
        motif_budget = int(cfg.get("candidate_motif_max_per_trace", len(ordered)))
        motif_budget = max(0, motif_budget)
        trap_budget = int(cfg.get("candidate_trap_max_per_trace", len(ordered)))
        trap_budget = max(0, trap_budget)

        pv_used = 0
        motif_used = 0
        trap_used = 0

        paragraphs = []
        for cand in ordered:
            include_pv = False
            if cfg.get("include_candidate_pv", True) and pv_used < pv_budget:
                base_prob = float(cfg.get("candidate_pv_prob", 0.40))
                if style == "concise":
                    base_prob *= 0.4
                elif style == "tactical":
                    base_prob *= 1.2
                if best_uci and getattr(cand, "uci", None) == best_uci:
                    base_prob *= float(cfg.get("candidate_pv_best_mult", 1.2))
                if rng.random() < min(max(base_prob, 0.0), 1.0):
                    include_pv = True
                    pv_used += 1

            include_motif = False
            if cfg.get("include_motifs", False) and motif_used < motif_budget:
                prob = float(cfg.get("candidate_motif_prob", 1.0))
                if style == "quick":
                    prob *= 0.25
                elif style == "concise":
                    prob *= 0.5
                elif style == "tactical":
                    prob *= 1.1
                if best_uci and getattr(cand, "uci", None) == best_uci:
                    prob *= float(cfg.get("candidate_motif_best_mult", 1.1))
                if rng.random() < min(max(prob, 0.0), 1.0):
                    include_motif = True
                    motif_used += 1

            include_trap = False
            if cfg.get("include_trap_detection", False) and trap_used < trap_budget:
                prob = float(cfg.get("candidate_trap_prob", 1.0))
                if style == "quick":
                    prob *= 0.25
                elif style == "concise":
                    prob *= 0.6
                elif style == "tactical":
                    prob *= 1.1
                if best_uci and getattr(cand, "uci", None) == best_uci:
                    prob *= float(cfg.get("candidate_trap_best_mult", 1.0))
                if rng.random() < min(max(prob, 0.0), 1.0):
                    include_trap = True
                    trap_used += 1

            paragraphs.append(
                self._analyze_candidate(
                    board,
                    cand,
                    best_win,
                    analysis,
                    style,
                    rng,
                    notation,
                    cfg,
                    include_motif=include_motif,
                    include_trap=include_trap,
                    include_pv=include_pv,
                )
            )
        return paragraphs

    def _analyze_candidate(
        self,
        board: chess.Board,
        cand: Any,
        best_win: float,
        analysis: Any,
        style: str,
        rng: random.Random,
        notation: str,
        cfg: Dict[str, Any],
        *,
        include_motif: bool = True,
        include_trap: bool = True,
        include_pv: bool = True,
    ) -> str:
        move_label = self._move_label(cand, notation)
        try:
            move = chess.Move.from_uci(cand.uci)
        except Exception:
            move = None
        features = self._move_features(board, move, cfg) if move else []
        idea = self._idea_phrase(features, rng, style)
        unclear_margin = float(cfg.get("unclear_win_margin", 0.04))
        quality = self._quality_phrase(cand, best_win, rng, style, unclear_margin)
        motif_line = ""
        if include_motif and cfg.get("include_motifs", False) and move is not None:
            motifs = self._detect_candidate_motifs(board, move, cand, cfg)
            motif_line = self._motif_phrase(motifs, rng)
        trap_line = ""
        if include_trap and cfg.get("include_trap_detection", False):
            trap = self._detect_trap_for_candidate(cand, analysis, cfg)
            trap_line = self._trap_phrase(trap, rng, board, cand, notation, cfg)

        pv_line = ""
        if include_pv and cfg.get("include_candidate_pv", True) and move is not None:
            pv_line = self._candidate_pv_snippet(board, cand, notation, cfg, rng)

        question_prob = float(cfg.get("candidate_question_prob", 0.35))
        if rng.random() < min(max(question_prob, 0.0), 1.0):
            intro = rng.choice([
                f"What about {move_label}?",
                f"Does {move_label} work here?",
                f"How about {move_label}?",
            ])
            if idea:
                idea = idea[:1].upper() + idea[1:]
        else:
            intro = rng.choice([
                f"Looking at {move_label},",
                f"For {move_label},",
                f"Considering {move_label},",
            ])

        parts = [intro, idea, pv_line, quality, motif_line, trap_line]
        return " ".join(part for part in parts if part)

    def _candidate_pv_snippet(
        self,
        board: chess.Board,
        cand: Any,
        notation: str,
        cfg: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        max_len = int(cfg.get("candidate_pv_max_len", 4))
        if max_len <= 0:
            return ""
        pv_uci = list(getattr(cand, "pv_uci", None) or [])
        if not pv_uci:
            return ""
        pruned_moves, _ = self._prune_pv_moves(board, pv_uci[:max_len], cfg)
        line = self._format_pv_line(board, pruned_moves, notation)
        if not line:
            return ""
        prefix = rng.choice([
            "One concrete line is:",
            "A sample line is:",
            "For example:",
        ])
        return f"{prefix} {line}."

    def _reconsideration_line(self, rng: random.Random) -> str:
        return rng.choice([
            "Actually, I should double-check the tactical details.",
            "On second thought, there may be a hidden resource here.",
            "Wait, there could be a subtle defense I missed.",
        ])

    def _dead_end_line(
        self,
        candidates: List[Any],
        best_analysis: Optional[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        if len(candidates) < 2:
            return ""
        worst = min(
            candidates,
            key=lambda c: getattr(c, "win_probability", 0.5),
        )
        move_label = self._move_label(worst, notation)
        reason = rng.choice([
            "it allows a strong reply",
            "it leaves the king exposed",
            "it drops material after a simple tactic",
            "it concedes too much activity",
        ])
        return f"I looked at {move_label}, but it does not work because {reason}."

    def _comparison_line(
        self,
        candidates: List[Any],
        best_analysis: Optional[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        if len(candidates) < 2:
            return ""
        ordered = sorted(
            candidates,
            key=lambda c: getattr(c, "win_probability", 0.5),
            reverse=True,
        )
        first = ordered[0]
        second = ordered[1]
        first_label = self._move_label(first, notation)
        second_label = self._move_label(second, notation)
        return rng.choice([
            f"Between {first_label} and {second_label}, the first looks more direct.",
            f"Comparing {first_label} vs {second_label}, the first seems more forcing.",
            f"{first_label} looks a bit cleaner than {second_label}.",
        ])

    def _conclusion_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        analysis: Any,
        decision_profile: Dict[str, Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        if best_analysis is None:
            return "I will choose a solid move and keep the position safe."
        best_move_label = self._move_label(best_analysis, notation)
        reason = self._idea_phrase(
            self._move_features(board, chess.Move.from_uci(best_analysis.uci), cfg),
            rng,
            style,
        )
        intro = rng.choice(PHRASE_LIBRARY["conclusion_intros"])
        format_pool = list(PHRASE_LIBRARY["conclusion_formats"])
        if decision_profile.get("forced_mate") or decision_profile.get("only_move"):
            format_pool = [
                "it has to be {move}.",
                "{move} is the answer.",
                "{move} is the right choice.",
                "{move} stands out as best.",
                "{move} is clearly best.",
                "{move} wins out.",
            ]
        elif decision_profile.get("many_options"):
            format_pool = [
                "I'm playing {move}.",
                "going with {move}.",
                "{move} is what I'd play.",
                "{move} makes the most sense.",
                "{move} gets the nod.",
                "{move} is the pick.",
            ]
        format_template = rng.choice(format_pool)
        intro_text = str(intro or "").strip()
        if intro_text.lower().endswith(" to") or intro_text.lower().endswith(" is"):
            conclusion = f"{intro_text} {best_move_label}."
        else:
            conclusion = f"{intro_text} {format_template.format(move=best_move_label)}"
        pv_line = ""
        pruned = False
        if reason:
            reason = reason[:1].upper() + reason[1:]
        confidence = self._confidence_clause(decision_profile, rng, cfg)
        if confidence:
            confidence = confidence.rstrip(".") + "."
        pv_prob = 0.0
        if cfg.get("include_pv", True):
            pv_prob = float(cfg.get("pv_prob", 1.0))
            if style == "quick":
                pv_prob *= 0.25
            elif style in ("concise", "dialogue"):
                pv_prob *= 0.6
            if decision_profile.get("forced_mate") or decision_profile.get("only_move"):
                pv_prob = max(pv_prob, float(cfg.get("pv_prob_forced_min", 0.85)))
            if rng.random() >= min(max(pv_prob, 0.0), 1.0):
                pv_prob = 0.0

        if pv_prob > 0.0:
            max_len = int(cfg.get("max_pv_length", 8))
            pv_uci = list(getattr(best_analysis, "pv_uci", None) or [])
            if pv_uci and max_len > 0:
                pruned_moves, pruned = self._prune_pv_moves(board, pv_uci[:max_len], cfg)
                line = self._format_pv_line(board, pruned_moves, notation)
                if line:
                    pv_line = f"Main line: {line}"
            elif notation == "san" and getattr(analysis, "best_pv", None):
                pv_moves, pruned = self._prune_pv_san(board, analysis.best_pv, cfg, max_len)
                if pv_moves:
                    pv_line = f"Main line: {' '.join(pv_moves)}"
            else:
                pv_uci_text, pruned = self._pv_line_uci(board, analysis, max_len, cfg)
                if pv_uci_text:
                    pv_line = f"Main line: {pv_uci_text}"
            if pv_line and pruned and cfg.get("pv_quiet_summary", True):
                quiet_summary = rng.choice([
                    "The line quiets down after that.",
                    "After that, the position stays quiet.",
                    "The rest of the line is quieter.",
                    "The position settles after that.",
                ])
                pv_line = f"{pv_line} {quiet_summary}"
        pv_highlights = self._pv_key_moments_line(board, best_analysis, analysis, style, rng, cfg, notation)
        pv_explainer = self._pv_explainer_block(board, best_analysis, style, rng, cfg, notation)
        head = " ".join(part for part in (conclusion, confidence, reason) if part).strip()
        if pv_line and pv_highlights:
            if pv_explainer:
                return f"{head}\n\n{pv_line}\n{pv_highlights}\n{pv_explainer}"
            return f"{head}\n\n{pv_line}\n{pv_highlights}"
        if pv_line:
            if pv_explainer:
                return f"{head}\n\n{pv_line}\n{pv_explainer}"
            return f"{head}\n\n{pv_line}"
        if pv_highlights:
            if pv_explainer:
                return f"{head}\n\n{pv_highlights}\n{pv_explainer}"
            return f"{head}\n\n{pv_highlights}"
        if pv_explainer:
            return f"{head}\n\n{pv_explainer}"
        return head

    def _confidence_clause(
        self,
        decision_profile: Dict[str, Any],
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if not bool(cfg.get("include_confidence_in_conclusion", True)):
            return ""

        if decision_profile.get("forced_mate"):
            return rng.choice(
                [
                    "This looks forcing, so calculation matters most",
                    "This is a forcing spot, so I'm confident in the concrete line",
                    "This is tactical and forcing, so the exact line matters",
                ]
            )
        if decision_profile.get("only_move"):
            return rng.choice(
                [
                    "This feels like a critical moment where accuracy matters",
                    "This looks close to an only-move situation, so precision matters",
                    "The margin for error is small here, so I want the most precise line",
                ]
            )
        if decision_profile.get("many_options"):
            return rng.choice(
                [
                    "Several moves look close, so I'm choosing the cleanest plan",
                    "There are multiple playable options, so I’ll take the most straightforward continuation",
                    "This isn't forced, so I’ll prefer the move with the clearest follow-up",
                ]
            )
        return rng.choice(
            [
                "This feels like the most consistent continuation",
                "This looks like the cleanest choice overall",
                "This seems like the most practical continuation",
            ]
        )

    def _pv_key_moments_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        analysis: Any,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        if not bool(cfg.get("include_pv_key_moments", True)):
            return ""

        prob = float(cfg.get("pv_key_moments_prob", 0.35))
        if style == "quick":
            prob *= 0.25
        elif style == "concise":
            prob *= 0.6
        if rng.random() >= min(max(prob, 0.0), 1.0):
            return ""

        if best_analysis is None:
            return ""
        pv_uci_full = list(getattr(best_analysis, "pv_uci", None) or [])
        if not pv_uci_full:
            return ""

        max_plies = int(cfg.get("pv_key_moments_max_plies", cfg.get("max_pv_length", 8)))
        max_plies = max(1, max_plies)
        pv_uci = pv_uci_full[:max_plies]

        temp = board.copy()
        labels: List[str] = []
        forcing: List[bool] = []
        motif: Optional[str] = None
        include_motif = bool(cfg.get("pv_key_moments_include_motif", True))

        for idx, uci in enumerate(pv_uci):
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in temp.legal_moves:
                break
            labels.append(self._move_notation(temp, move, notation))
            is_forcing = temp.is_capture(move) or temp.gives_check(move) or bool(move.promotion)
            forcing.append(bool(is_forcing))

            if include_motif and motif is None and bool(cfg.get("include_motifs", True)):
                try:
                    fake = type("PVMove", (), {})()
                    fake.uci = uci
                    fake.pv_uci = pv_uci_full[idx:]
                    motifs = self._detect_candidate_motifs(temp, move, fake, cfg, rng=rng)
                    if motifs:
                        motif = motifs[0]
                except Exception:
                    motif = None

            temp.push(move)

        if not labels:
            return ""

        plies = len(labels)
        forcing_count = sum(1 for f in forcing if f)

        first_forcing_idx = next((i for i, f in enumerate(forcing) if f), None)

        quiet_streak = 0
        settle_idx: Optional[int] = None
        forcing_seen = False
        for i, f in enumerate(forcing):
            if f:
                forcing_seen = True
                quiet_streak = 0
            else:
                quiet_streak += 1
                if forcing_seen and quiet_streak >= 2:
                    settle_idx = i
                    break

        parts: List[str] = []
        if bool(cfg.get("pv_key_moments_include_forcingness", True)):
            parts.append(
                f"Forcingness: {forcing_count}/{plies} plies are checks/captures/promotions."
            )
        if first_forcing_idx is not None:
            parts.append(f"First forcing point: {labels[first_forcing_idx]}.")
        if settle_idx is not None and settle_idx - 1 >= 0 and settle_idx - 1 < len(labels):
            parts.append(f"After {labels[settle_idx - 1]}, the line becomes quieter.")
        if motif:
            parts.append(self._motif_phrase([motif], rng))

        if not parts:
            return ""

        prefix = rng.choice(
            [
                "PV highlights:",
                "Main-line highlights:",
                "Key moments in the PV:",
                "PV key moments:",
            ]
        )
        return f"{prefix} " + " ".join(p for p in parts if p).strip()

    def _pv_explainer_block(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """
        Explain the best PV in plain English using lightweight move heuristics.

        This uses only already-computed PV moves; it does not require additional
        Stockfish calls.
        """
        if not bool(cfg.get("include_pv_explainer", True)):
            return ""

        prob = float(cfg.get("pv_explainer_prob", 0.25))
        if style == "quick":
            prob *= 0.25
        elif style == "concise":
            prob *= 0.6
        if rng.random() >= min(max(prob, 0.0), 1.0):
            return ""

        if best_analysis is None:
            return ""

        pv_uci_full = list(getattr(best_analysis, "pv_uci", None) or [])
        if not pv_uci_full:
            return ""

        max_plies = int(cfg.get("pv_explainer_max_plies", 6))
        max_plies = max(1, max_plies)
        max_features = int(cfg.get("pv_explainer_max_features_per_move", 2))
        max_features = max(0, max_features)
        fmt = str(cfg.get("pv_explainer_format", "bullets")).strip().lower()

        temp = board.copy()
        entries: List[Tuple[str, str]] = []
        for uci in pv_uci_full[:max_plies]:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in temp.legal_moves:
                break
            label = self._move_notation(temp, move, notation)
            features = self._move_features(temp, move, cfg)
            if max_features > 0:
                features = features[:max_features]
            if features:
                if len(features) == 1:
                    desc = features[0]
                elif len(features) == 2:
                    desc = f"{features[0]} and {features[1]}"
                else:
                    desc = ", ".join(features)
            else:
                desc = rng.choice(
                    [
                        "improves piece activity",
                        "stabilizes the position",
                        "keeps the position under control",
                        "improves coordination",
                    ]
                )
            entries.append((label, desc))
            temp.push(move)

        if not entries:
            return ""

        if fmt == "sentences":
            chunks = [f"{label} ({desc})" for label, desc in entries]
            return f"PV explained: " + "; ".join(chunks) + "."

        lines = ["PV explained:"]
        for label, desc in entries:
            lines.append(f"- {label}: {desc}.")
        return "\n".join(lines)

    # =========================================================================
    # NEW STYLE GENERATORS
    # =========================================================================

    def _generate_quick_style(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """Generate a very brief 2-3 sentence trace."""
        side = "White" if board.turn == chess.WHITE else "Black"
        phase = self._position_phase(board)

        # Sentence 1: Brief position description
        opener = rng.choice([
            f"{phase.title()} position. {side} to move.",
            f"{side} to move in this {phase}.",
            f"Quick look: {phase}, {side}'s turn.",
            f"{phase.title()}. {side} is up.",
            f"Simple {phase} situation. {side} to play.",
        ])

        # Sentence 2: Quick candidates with one-line ideas
        if candidates:
            top_moves = candidates[:3]
            move_summaries = []
            for cand in top_moves:
                label = self._move_label(cand, notation)
                idea = self._quick_idea(board, cand, rng)
                move_summaries.append(f"{label} ({idea})")
            candidates_text = rng.choice([
                f"Main ideas: {', '.join(move_summaries)}.",
                f"Options: {', '.join(move_summaries)}.",
                f"Candidates: {', '.join(move_summaries)}.",
            ])
        else:
            candidates_text = "Looking for the best continuation."

        # Sentence 3: Quick conclusion with optional motif
        if best_analysis:
            best_label = self._move_label(best_analysis, notation)
            motif_text = ""
            if cfg.get("include_motifs", False):
                try:
                    move = chess.Move.from_uci(best_analysis.uci)
                    motifs = self._detect_candidate_motifs(board, move, best_analysis, cfg)
                    if motifs:
                        motif_text = f" ({motifs[0]})"
                except Exception:
                    pass
            conclusion = rng.choice([
                f"Best: {best_label}{motif_text}.",
                f"Play {best_label}{motif_text}.",
                f"The move is {best_label}{motif_text}.",
                f"Go with {best_label}{motif_text}.",
                f"{best_label} is right{motif_text}.",
            ])
        else:
            conclusion = "Solid development is key."

        return f"{opener} {candidates_text} {conclusion}"

    def _quick_idea(self, board: chess.Board, cand: Any, rng: random.Random) -> str:
        """Generate a 1-3 word idea for quick style."""
        try:
            move = chess.Move.from_uci(cand.uci)
        except Exception:
            return rng.choice(PHRASE_LIBRARY["quick_ideas"])

        if board.gives_check(move):
            return rng.choice(["check", "checks", "gives check"])
        if board.is_capture(move):
            return rng.choice(["captures", "takes", "wins material"])
        if board.is_castling(move):
            return rng.choice(["castles", "safety", "king safety"])
        if move.promotion:
            return rng.choice(["promotes", "queening", "promotion"])

        return rng.choice(PHRASE_LIBRARY["quick_ideas"])

    def _move_type_phrase(
        self,
        board: chess.Board,
        move: chess.Move,
        rng: random.Random,
    ) -> str:
        """Get a descriptive phrase based on move type (capture, check, or quiet)."""
        if board.gives_check(move):
            return rng.choice(PHRASE_LIBRARY["check_phrases"])
        if board.is_capture(move):
            return rng.choice(PHRASE_LIBRARY["capture_phrases"])
        return rng.choice(PHRASE_LIBRARY["quiet_move_phrases"])

    def _generate_problem_focused_style(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """Generate trace as: problem -> failed attempts -> solution."""
        sections = []

        # 1. Problem Identification
        problem = self._identify_problem(board, best_analysis, rng)
        sections.append(problem)

        # 2. Failed Attempts (show 1-2 inferior moves)
        if len(candidates) >= 2:
            failed = self._failed_attempts(board, candidates, best_analysis, rng, notation)
            if failed:
                sections.append(failed)

        # 3. Successful Attempt (with optional motif)
        if best_analysis:
            success = self._successful_attempt(board, best_analysis, rng, notation, cfg)
            sections.append(success)

        # 4. Brief Conclusion
        if best_analysis:
            best_label = self._move_label(best_analysis, notation)
            conclusion = rng.choice([
                f"Therefore, {best_label} solves the position.",
                f"The answer is {best_label}.",
                f"{best_label} is the key move that addresses everything.",
                f"So {best_label} is correct.",
                f"{best_label} handles the situation.",
            ])
            sections.append(conclusion)

        return "\n\n".join(sections)

    def _identify_problem(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        rng: random.Random,
    ) -> str:
        """Identify the main challenge in the position."""
        problems = []

        if board.is_check():
            problems.append("must escape check")

        hanging = self._find_hanging_pieces_for_color(board, board.turn, 1)
        if hanging:
            piece, square = hanging[0]
            name = PIECE_NAMES.get(piece.piece_type, "piece")
            problems.append(f"the {name} on {chess.square_name(square)} is hanging")

        if self._is_exposed_king(board, board.turn):
            problems.append("king safety is concerning")

        # Check for opponent threats
        opp_checks = self._count_opponent_checks(board)
        if opp_checks > 0:
            problems.append("opponent has checking ideas")

        if not problems:
            problems.append("finding the most active continuation")

        problem_text = problems[0] if len(problems) == 1 else " and ".join(problems[:2])

        intro = rng.choice(PHRASE_LIBRARY["problem_intros"])
        return f"{intro} {problem_text}."

    def _count_opponent_checks(self, board: chess.Board) -> int:
        """Count how many checking moves opponent has."""
        opp_board = board.copy()
        opp_board.turn = not board.turn
        return sum(1 for move in opp_board.legal_moves if opp_board.gives_check(move))

    def _failed_attempts(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        """Show 1-2 moves that don't work and why."""
        best_uci = best_analysis.uci if best_analysis else ""
        inferior = [c for c in candidates if c.uci != best_uci]

        if not inferior:
            return ""

        attempts = []
        for i, cand in enumerate(inferior[:2]):
            label = self._move_label(cand, notation)

            if i == 0:
                prefix = rng.choice(PHRASE_LIBRARY["failed_attempt_intros"]).format(move=label)
            else:
                prefix = rng.choice(PHRASE_LIBRARY["failed_attempt_second"]).format(move=label)

            failure_reason = self._why_move_fails(cand, rng)
            attempts.append(f"{prefix} {failure_reason}")

        return " ".join(attempts)

    def _why_move_fails(self, cand: Any, rng: random.Random) -> str:
        """Generate reason why a move doesn't work."""
        win_prob = getattr(cand, "win_probability", 0.5)

        if win_prob < 0.35:
            return rng.choice(PHRASE_LIBRARY["failure_reasons_bad"])
        elif win_prob < 0.45:
            return rng.choice(PHRASE_LIBRARY["failure_reasons_ok"])
        else:
            return rng.choice(PHRASE_LIBRARY["failure_reasons_close"])

    def _successful_attempt(
        self,
        board: chess.Board,
        best_analysis: Any,
        rng: random.Random,
        notation: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Show the move that works and why."""
        cfg = cfg or {}
        label = self._move_label(best_analysis, notation)

        try:
            move = chess.Move.from_uci(best_analysis.uci)
            features = self._move_features(board, move, cfg)
        except Exception:
            move = None
            features = []

        # Check for motifs
        motif_text = ""
        if cfg.get("include_motifs", False) and move is not None:
            try:
                motifs = self._detect_candidate_motifs(board, move, best_analysis, cfg)
                if motifs:
                    motif_text = f" It's a {motifs[0]}."
            except Exception:
                pass

        # Generate reason based on features or move type
        if features:
            feature_text = ", ".join(features[:2])
            reason = f"It {feature_text}."
        elif move is not None:
            # Use move-type specific phrase
            reason = self._move_type_phrase(board, move, rng)
        else:
            reason = rng.choice([
                "It addresses all the key concerns.",
                "This handles everything cleanly.",
                "It solves the position's demands.",
                "This is the most accurate.",
                "It keeps things under control.",
            ])

        intro = rng.choice(PHRASE_LIBRARY["solution_intros"]).format(move=label)
        return f"{intro}{motif_text} {reason}"

    def _generate_intuition_style(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """Generate trace as: gut feeling -> verification -> conclusion."""
        sections = []

        # 1. First Impression / Gut Feeling
        first_impression = rng.choice(PHRASE_LIBRARY["intuition_first_impressions"])
        sections.append(f"First impression: {first_impression}")

        # 2. Initial Move Preference (may or may not be best)
        initial_guess_label = ""
        if candidates:
            # Sometimes pick the best, sometimes a close second
            if rng.random() < 0.7 and best_analysis:
                guess = best_analysis
            else:
                guess = rng.choice(candidates)
            initial_guess_label = self._move_label(guess, notation)
            preference = rng.choice(PHRASE_LIBRARY["intuition_preference_intros"]).format(
                move=initial_guess_label
            )
            sections.append(preference)

        # 3. Verification with Concrete Lines and optional motif
        if best_analysis:
            best_label = self._move_label(best_analysis, notation)
            verify_intro = rng.choice(PHRASE_LIBRARY["intuition_verify"]).format(move=best_label)

            pv_text = ""
            if cfg.get("include_pv", True):
                max_len = min(cfg.get("max_pv_length", 8), 5)
                pv_uci, _ = self._pv_line_uci(board, analysis, max_len, cfg)
                if pv_uci:
                    pv_text = f" The line goes: {pv_uci}."

            motif_text = ""
            if cfg.get("include_motifs", False):
                try:
                    move = chess.Move.from_uci(best_analysis.uci)
                    motifs = self._detect_candidate_motifs(board, move, best_analysis, cfg)
                    if motifs:
                        motif_text = f" This is a {motifs[0]}."
                except Exception:
                    pass

            verification = f"{verify_intro}{pv_text}{motif_text} This looks sound."
            sections.append(verification)

        # 4. Confirmation or Adjustment
        if best_analysis:
            best_label = self._move_label(best_analysis, notation)
            # Check if initial preference matched
            was_correct = initial_guess_label == best_label

            if was_correct:
                conclusion = rng.choice(PHRASE_LIBRARY["intuition_confirmed"]).format(
                    move=best_label
                )
            else:
                conclusion = rng.choice(PHRASE_LIBRARY["intuition_adjusted"]).format(
                    move=best_label
                )
            sections.append(conclusion)

        return "\n\n".join(sections)

    def _generate_comparison_style(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        """Generate trace framed as comparison between candidates."""
        if len(candidates) < 2:
            # Fall back to quick style if not enough candidates
            return self._generate_quick_style(
                board, candidates, best_analysis, analysis, rng, cfg, notation
            )

        sections = []

        # 1. Frame the Decision
        frame = self._frame_decision(candidates, rng, notation)
        sections.append(frame)

        # 2. Side by Side Comparison
        comparison = self._side_by_side(candidates[:3], best_analysis, rng, notation)
        sections.append(comparison)

        # 3. What Makes One Better (with optional motif)
        if best_analysis:
            differentiator = self._key_difference(best_analysis, rng, notation)
            if cfg.get("include_motifs", False):
                try:
                    move = chess.Move.from_uci(best_analysis.uci)
                    motifs = self._detect_candidate_motifs(board, move, best_analysis, cfg)
                    if motifs:
                        differentiator += f" (a {motifs[0]})"
                except Exception:
                    pass
            sections.append(differentiator)

        # 4. Final Choice
        if best_analysis:
            label = self._move_label(best_analysis, notation)
            conclusion = rng.choice(PHRASE_LIBRARY["comparison_conclusions"]).format(move=label)
            sections.append(conclusion)

        return "\n\n".join(sections)

    def _frame_decision(
        self,
        candidates: List[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        """Frame the main decision between candidates."""
        if len(candidates) >= 3:
            labels = [self._move_label(c, notation) for c in candidates[:3]]
            template = rng.choice(PHRASE_LIBRARY["comparison_frames_three"])
            return template.format(m1=labels[0], m2=labels[1], m3=labels[2])
        else:
            labels = [self._move_label(c, notation) for c in candidates[:2]]
            template = rng.choice(PHRASE_LIBRARY["comparison_frames_two"])
            return template.format(m1=labels[0], m2=labels[1])

    def _side_by_side(
        self,
        candidates: List[Any],
        best_analysis: Optional[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        """Generate side-by-side pros/cons for each candidate."""
        comparisons = []

        for cand in candidates:
            label = self._move_label(cand, notation)
            pros = self._get_pros(cand, rng)
            cons = self._get_cons(cand, best_analysis, rng)
            comparisons.append(f"{label}: {pros} / {cons}")

        return "\n".join(comparisons)

    def _get_pros(self, cand: Any, rng: random.Random) -> str:
        """Generate pros for a move."""
        win_prob = getattr(cand, "win_probability", 0.5)

        if win_prob > 0.55:
            return rng.choice(PHRASE_LIBRARY["comparison_pros_strong"])
        elif win_prob > 0.45:
            return rng.choice(PHRASE_LIBRARY["comparison_pros_solid"])
        else:
            return rng.choice(PHRASE_LIBRARY["comparison_pros_fighting"])

    def _get_cons(
        self,
        cand: Any,
        best_analysis: Optional[Any],
        rng: random.Random,
    ) -> str:
        """Generate cons for a move."""
        if best_analysis and cand.uci == best_analysis.uci:
            return rng.choice(PHRASE_LIBRARY["comparison_cons_best"])

        win_prob = getattr(cand, "win_probability", 0.5)
        best_prob = getattr(best_analysis, "win_probability", 0.5) if best_analysis else 0.5

        if win_prob < best_prob - 0.1:
            return rng.choice(PHRASE_LIBRARY["comparison_cons_inferior"])
        elif win_prob < best_prob - 0.05:
            return rng.choice(PHRASE_LIBRARY["comparison_cons_close"])
        else:
            return rng.choice(PHRASE_LIBRARY["comparison_cons_marginal"])

    def _key_difference(
        self,
        best_analysis: Any,
        rng: random.Random,
        notation: str,
    ) -> str:
        """Explain what makes the best move stand out."""
        best_label = self._move_label(best_analysis, notation)
        template = rng.choice(PHRASE_LIBRARY["comparison_key_difference"])
        return template.format(move=best_label)

    # =========================================================================
    # END NEW STYLE GENERATORS
    # =========================================================================

    def _move_label(self, cand: Any, notation: str) -> str:
        if notation == "san":
            return getattr(cand, "san", getattr(cand, "uci", ""))
        return getattr(cand, "uci", getattr(cand, "san", ""))

    def _move_features(
        self,
        board: chess.Board,
        move: Optional[chess.Move],
        cfg: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if move is None:
            return []
        cfg = cfg or {}
        features = []
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)
        is_castle = board.is_castling(move)

        if is_capture:
            captured = board.piece_at(move.to_square)
            if captured:
                features.append(f"captures the {PIECE_NAMES.get(captured.piece_type, 'piece')}")
            else:
                features.append("captures material")
        if is_check:
            features.append("gives check")
        if is_castle:
            features.append("castles to safety")
        if move.promotion:
            features.append("promotes a pawn")
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if chess.square_rank(move.from_square) in (0, 7):
                name = PIECE_NAMES.get(piece.piece_type, "piece")
                features.append(f"develops the {name}")
        if piece and piece.piece_type == chess.PAWN:
            if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                features.append(f"claims central space on {chess.square_name(move.to_square)}")

        is_quiet = not (is_capture or is_check or is_castle or bool(move.promotion))
        if is_quiet and piece is not None:
            if not board.is_legal(move):
                return features
            board_after = board.copy(stack=False)
            board_after.push(move)
            attacked = board_after.attacks(move.to_square)

            # Mention concrete targets (especially when a quiet move hits a major piece).
            if bool(cfg.get("feature_include_target_piece", True)):
                min_value = int(cfg.get("feature_target_min_value", 3))
                targets: List[Tuple[int, int, chess.Square]] = []
                for sq in attacked:
                    target_piece = board_after.piece_at(sq)
                    if not target_piece or target_piece.color == piece.color:
                        continue
                    value = PIECE_VALUES.get(target_piece.piece_type, 0)
                    targets.append((value, target_piece.piece_type, sq))
                targets.sort(key=lambda x: x[0], reverse=True)
                if targets and targets[0][0] >= min_value:
                    _, piece_type, sq = targets[0]
                    features.append(
                        f"targets the {PIECE_NAMES.get(piece_type, 'piece')} on {chess.square_name(sq)}"
                    )

            # Rooks: file activity is a common human cue.
            if piece.piece_type == chess.ROOK and bool(cfg.get("feature_include_rook_files", True)):
                file_idx = chess.square_file(move.to_square)
                file_letter = "abcdefgh"[file_idx]
                if file_idx in self._open_files(board):
                    features.append(f"puts a rook on the open {file_letter}-file")
                elif file_idx in self._half_open_files(board, piece.color):
                    features.append(f"puts a rook on the half-open {file_letter}-file")

            # Knights: "square talk" (posts, outposts, central pressure).
            if piece.piece_type == chess.KNIGHT and bool(cfg.get("feature_include_knight_posts", True)):
                to_sq = move.to_square
                if to_sq in WEAK_SQUARE_CANDIDATES or self._is_outpost_square(board_after, piece.color, to_sq):
                    features.append(f"posts a knight on {chess.square_name(to_sq)}")

            # Mention key-square pressure (central/weak squares) when the moved piece attacks one.
            if bool(cfg.get("feature_include_square_pressure", True)):
                key_sq = next(
                    (sq for sq in WEAK_SQUARE_CANDIDATES if sq in attacked and sq != move.to_square),
                    None,
                )
                if key_sq is None:
                    weak_squares = self._find_weak_squares(board_after, piece.color)
                    key_sq = next(
                        (sq for sq in weak_squares if sq in attacked and sq != move.to_square),
                        None,
                    )
                if key_sq is not None:
                    features.append(f"challenges the {chess.square_name(key_sq)} square")
        return features

    def _is_outpost_square(self, board: chess.Board, color: bool, square: chess.Square) -> bool:
        """Return True if a square is pawn-supported and cannot be chased by enemy pawns."""
        rank_idx = chess.square_rank(square)
        # Restrict to squares in the opponent's half (typical "outpost" usage).
        if color == chess.WHITE and rank_idx < 4:
            return False
        if color == chess.BLACK and rank_idx > 3:
            return False
        if not self._pawn_supported(board, color, square):
            return False
        return not self._pawn_can_attack_square(board, not color, square)

    @staticmethod
    def _pawn_supported(board: chess.Board, color: bool, square: chess.Square) -> bool:
        for attacker in board.attackers(color, square):
            piece = board.piece_at(attacker)
            if piece and piece.piece_type == chess.PAWN:
                return True
        return False

    @staticmethod
    def _pawn_can_attack_square(board: chess.Board, pawn_color: bool, square: chess.Square) -> bool:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        # If a pawn of `pawn_color` is on either "capturing-from" square, it attacks `square`.
        from_rank = rank_idx - 1 if pawn_color == chess.WHITE else rank_idx + 1
        if from_rank < 0 or from_rank > 7:
            return False
        for df in (-1, 1):
            from_file = file_idx + df
            if from_file < 0 or from_file > 7:
                continue
            from_sq = chess.square(from_file, from_rank)
            piece = board.piece_at(from_sq)
            if piece and piece.color == pawn_color and piece.piece_type == chess.PAWN:
                return True
        return False

    def _idea_phrase(self, features: List[str], rng: random.Random, style: str) -> str:
        if not features:
            if style == "tactical":
                return rng.choice([
                    "it keeps the initiative and creates immediate threats.",
                    "it forces a reply and keeps pressure on the opponent.",
                    "it keeps the attack moving.",
                ])
            if style == "concise":
                return rng.choice([
                    "it keeps things simple.",
                    "it is a safe, flexible choice.",
                    "it is a solid move.",
                ])
            return rng.choice([
                "it improves coordination and keeps options open.",
                "it keeps the position flexible.",
                "it develops sensibly without committing too much.",
            ])
        if len(features) == 1:
            base = f"it {features[0]}."
            if style == "tactical":
                tail = rng.choice([
                    "This keeps the initiative.",
                    "This is forcing.",
                    "This keeps pressure on.",
                ])
                return f"{base} {tail}"
            return base
        if len(features) == 2:
            base = f"it {features[0]} and {features[1]}."
        else:
            base = f"it {features[0]}, {features[1]}, and {features[2]}."
        if style == "tactical":
            tail = rng.choice([
                "This keeps the initiative.",
                "This is the forcing option.",
                "This keeps momentum.",
            ])
            return f"{base} {tail}"
        if style == "concise":
            return base
        return base

    def _quality_phrase(
        self,
        cand: Any,
        best_win: float,
        rng: random.Random,
        style: str,
        unclear_margin: float,
    ) -> str:
        mate_in = getattr(cand, "mate_in", None)
        if mate_in is not None:
            if mate_in > 0:
                return f"This is a forced mate in {mate_in}."
            return f"This allows mate in {abs(mate_in)}."
        win_prob = getattr(cand, "win_probability", 0.5)
        delta = best_win - win_prob
        if abs(win_prob - 0.5) <= unclear_margin:
            if delta <= 0.02:
                return rng.choice([
                    "It looks like one of several reasonable options.",
                    "It is a fine choice in an equal position.",
                    "This is a sensible choice in a balanced position.",
                ])
            if delta <= 0.06:
                return rng.choice([
                    "It is a reasonable alternative in a balanced position.",
                    "It seems playable with equality.",
                    "It keeps the game about equal.",
                ])
        if delta <= 0.02:
            if style == "tactical":
                return rng.choice([
                    "This looks most forcing.",
                    "This keeps the initiative most cleanly.",
                    "This feels like the sharpest line.",
                ])
            return rng.choice([
                "It looks strongest.",
                "This seems like the front-runner.",
                "It feels like the top choice.",
            ])
        if delta <= 0.06:
            if style == "concise":
                return rng.choice([
                    "It looks fine.",
                    "This is playable.",
                    "It seems solid.",
                ])
            return rng.choice([
                "It looks very solid.",
                "This is a strong alternative.",
                "It seems quite playable.",
            ])
        if delta <= 0.12:
            if style == "tactical":
                return rng.choice([
                    "It is playable, but less forcing.",
                    "It gives chances but lacks bite.",
                    "It keeps play going, but is not the sharpest.",
                ])
            return rng.choice([
                "It is a reasonable try.",
                "It seems playable but not perfect.",
                "It is usable but less forcing.",
            ])
        return rng.choice([
            "It looks risky in comparison.",
            "It seems inferior to the leading option.",
            "It is probably not the most accurate.",
        ])

    def _append_positional_cues(
        self,
        base: str,
        board: chess.Board,
        cfg: Dict[str, Any],
    ) -> str:
        if not cfg.get("include_positional_cues", False):
            return base
        cues = self._positional_cues(board, cfg)
        if not cues:
            return base
        return f"{base} Positional notes: {', '.join(cues)}."

    def _positional_cues(
        self,
        board: chess.Board,
        cfg: Dict[str, Any],
    ) -> List[str]:
        cues: List[str] = []
        max_cues = int(cfg.get("max_positional_cues", 2))
        color = board.turn

        bishops = board.pieces(chess.BISHOP, color)
        if len(bishops) >= 2:
            cues.append("bishop pair")

        open_files = self._open_files(board)
        rooks = board.pieces(chess.ROOK, color)
        for square in rooks:
            file_idx = chess.square_file(square)
            if file_idx in open_files:
                file_letter = "abcdefgh"[file_idx]
                cues.append(f"rook on open {file_letter}-file")
                break

        weak_squares = self._find_weak_squares(board, color)
        if weak_squares:
            cues.append(f"weak square on {chess.square_name(weak_squares[0])}")

        passed_pawn = self._find_passed_pawn(board, color)
        if passed_pawn is not None:
            cues.append(f"passed pawn on {chess.square_name(passed_pawn)}")

        outpost = self._find_outpost(board, color)
        if outpost is not None:
            cues.append(f"outpost on {chess.square_name(outpost)}")

        isolated = self._find_isolated_pawn(board, color)
        if isolated is not None:
            cues.append(f"isolated pawn on {chess.square_name(isolated)}")

        doubled_file = self._find_doubled_pawn_file(board, color)
        if doubled_file is not None:
            file_letter = "abcdefgh"[doubled_file]
            cues.append(f"doubled pawns on {file_letter}-file")

        backward = self._find_backward_pawn(board, color)
        if backward is not None:
            cues.append(f"backward pawn on {chess.square_name(backward)}")

        half_open = self._half_open_files(board, color)
        if half_open:
            file_letter = "abcdefgh"[half_open[0]]
            cues.append(f"half-open {file_letter}-file")

        majority = self._pawn_majority(board, color)
        if majority:
            cues.append(majority)

        seventh = self._rook_on_seventh(board, color)
        if seventh:
            cues.append(seventh)

        if self._is_exposed_king(board, color):
            cues.append("exposed king")

        return cues[:max_cues]

    def _position_phase(self, board: chess.Board) -> str:
        piece_count = len(board.piece_map())
        if piece_count <= 10:
            return "endgame"
        if board.fullmove_number <= 10:
            return "opening"
        return "middlegame"

    def _find_move(self, analysis: Any, uci: str) -> Optional[Any]:
        if not uci:
            return None
        for move in analysis.move_analyses:
            if move.uci == uci:
                return move
        return None

    def _pv_line_uci(
        self,
        board: chess.Board,
        analysis: Any,
        max_len: int,
        cfg: Dict[str, Any],
    ) -> Tuple[str, bool]:
        if not analysis.best_pv:
            return "", False
        temp = board.copy()
        uci_moves = []
        for san in analysis.best_pv[:max_len]:
            try:
                move = temp.parse_san(san)
            except Exception:
                break
            uci_moves.append(move.uci())
            temp.push(move)
        pruned, was_pruned = self._prune_pv_moves(board, uci_moves, cfg)
        return " ".join(pruned), was_pruned

    def _prune_pv_san(
        self,
        board: chess.Board,
        pv_san: List[str],
        cfg: Dict[str, Any],
        max_len: int,
    ) -> Tuple[List[str], bool]:
        if max_len > 0:
            pv_san = pv_san[:max_len]
        if not cfg.get("pv_prune_quiet", True):
            return pv_san, False
        max_quiet = int(cfg.get("pv_prune_quiet_plies", 2))
        min_moves = int(cfg.get("pv_prune_min_moves", 2))
        temp = board.copy()
        quiet = 0
        filtered: List[str] = []
        for san in pv_san:
            try:
                move = temp.parse_san(san)
            except Exception:
                break
            forcing = self._is_forcing_move(temp, move)
            filtered.append(san)
            temp.push(move)
            if forcing:
                quiet = 0
            else:
                quiet += 1
            if quiet >= max_quiet and len(filtered) >= min_moves:
                return filtered, True
        return filtered, False

    def _prune_pv_moves(
        self,
        board: chess.Board,
        pv_uci: List[str],
        cfg: Dict[str, Any],
    ) -> Tuple[List[str], bool]:
        if not cfg.get("pv_prune_quiet", True):
            return pv_uci, False
        max_quiet = int(cfg.get("pv_prune_quiet_plies", 2))
        min_moves = int(cfg.get("pv_prune_min_moves", 2))
        temp = board.copy()
        quiet = 0
        filtered: List[str] = []
        for uci in pv_uci:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in temp.legal_moves:
                break
            forcing = self._is_forcing_move(temp, move)
            filtered.append(uci)
            temp.push(move)
            if forcing:
                quiet = 0
            else:
                quiet += 1
            if quiet >= max_quiet and len(filtered) >= min_moves:
                return filtered, True
        return filtered, False

    @staticmethod
    def _is_forcing_move(board: chess.Board, move: chess.Move) -> bool:
        return board.is_capture(move) or board.gives_check(move) or bool(move.promotion)

    def _select_style(
        self,
        cfg: Dict[str, Any],
        source: Optional[str],
        rng: random.Random,
        *,
        position: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Explicit style override
        if cfg.get("style"):
            return cfg["style"]

        # Use configured weights if provided
        weights_raw = cfg.get("style_weights")
        if weights_raw:
            weights = {style: float(weights_raw.get(style, 0.0)) for style in STYLE_CONFIGS}
            if source == "puzzle" and position is not None:
                weights = self._apply_puzzle_style_bias(weights, position, cfg)
            return self._weighted_choice(weights, rng)

        # Source-based defaults for puzzles
        if source == "puzzle":
            puzzle_weights = {
                "tactical": 0.35,
                "problem_focused": 0.30,
                "quick": 0.15,
                "intuition": 0.10,
                "comparison_focused": 0.10,
            }
            return self._weighted_choice(puzzle_weights, rng)

        # Default weighted selection using STYLE_CONFIGS
        default_weights = {style: config["default_weight"] for style, config in STYLE_CONFIGS.items()}
        return self._weighted_choice(default_weights, rng)

    def _maybe_override_style(
        self,
        style: str,
        decision_profile: Dict[str, Any],
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if cfg.get("style"):
            return style
        if not bool(cfg.get("dynamic_style_bias", True)):
            return style

        prob = float(cfg.get("dynamic_style_bias_prob", 0.35))
        if rng.random() >= min(max(prob, 0.0), 1.0):
            return style

        if decision_profile.get("forced_mate"):
            return rng.choice(["tactical", "problem_focused", "thorough"])

        if decision_profile.get("only_move"):
            return rng.choice(["problem_focused", "tactical", style])

        if decision_profile.get("many_options"):
            return rng.choice(["comparison_focused", "intuition", style])

        return style

    def _apply_puzzle_style_bias(
        self,
        weights: Dict[str, float],
        position: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, float]:
        if not bool(cfg.get("puzzle_style_bias_enabled", True)):
            return weights

        themes = position.get("puzzle_themes") or []
        if isinstance(themes, str):
            themes = [t for t in themes.split() if t]
        themes_lc = {str(t).strip().lower() for t in themes if t}
        rating = position.get("puzzle_rating")
        try:
            rating_int = int(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating_int = None

        adjusted = dict(weights)

        def bump(style: str, factor: float) -> None:
            adjusted[style] = max(0.0, float(adjusted.get(style, 0.0)) * float(factor))

        if any("endgame" in t for t in themes_lc):
            bump("thorough", 1.25)
            bump("concise", 1.15)
            bump("intuition", 1.15)
            bump("tactical", 0.85)
            bump("problem_focused", 0.90)

        if any("mate" in t for t in themes_lc):
            bump("tactical", 1.45)
            bump("problem_focused", 1.25)
            bump("thorough", 1.15)
            bump("quick", 0.90)

        tactical_markers = (
            "fork",
            "pin",
            "skewer",
            "discovered",
            "double",
            "sacrifice",
            "deflection",
            "attraction",
            "interference",
            "clearance",
            "xray",
            "hanging",
            "trapped",
            "backrank",
        )
        if any(any(marker in t for marker in tactical_markers) for t in themes_lc):
            bump("tactical", 1.20)
            bump("problem_focused", 1.10)

        if "opening" in themes_lc:
            bump("quick", 1.20)
            bump("concise", 1.15)
            bump("thorough", 0.85)

        if "middlegame" in themes_lc:
            bump("thorough", 1.05)

        if "long" in themes_lc:
            bump("thorough", 1.10)
            bump("comparison_focused", 1.10)
            bump("quick", 0.90)

        if rating_int is not None:
            if rating_int >= 2200:
                bump("thorough", 1.20)
                bump("comparison_focused", 1.15)
                bump("quick", 0.70)
            elif rating_int <= 1200:
                bump("quick", 1.20)
                bump("concise", 1.10)
                bump("thorough", 0.85)

        return adjusted

    def _decision_profile(
        self,
        analysis: Any,
        best_analysis: Optional[Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        best_move_uci = getattr(best_analysis, "uci", None)

        move_probs = dict(getattr(analysis, "move_probs", None) or {})
        probs = [float(p) for p in move_probs.values() if p is not None and float(p) > 0.0]
        n_probs = len(probs)
        entropy = 0.0
        if n_probs:
            entropy = -sum(p * math.log(p) for p in probs)
        entropy_norm = entropy / math.log(n_probs) if n_probs > 1 else 0.0
        effective_moves = math.exp(entropy) if n_probs else 0.0

        best_prob = float(move_probs.get(str(best_move_uci), 0.0)) if best_move_uci else 0.0

        cp_gap = None
        second_cp_loss = None
        move_analyses = list(getattr(analysis, "move_analyses", None) or [])
        if best_analysis is not None and len(move_analyses) >= 2:
            ordered = sorted(move_analyses, key=lambda m: getattr(m, "centipawn", 0), reverse=True)
            best = next((m for m in ordered if getattr(m, "uci", None) == best_analysis.uci), ordered[0])
            second = next((m for m in ordered if getattr(m, "uci", None) != getattr(best, "uci", None)), None)
            if second is not None:
                cp_gap = int(getattr(best, "centipawn", 0)) - int(getattr(second, "centipawn", 0))
                second_cp_loss = int(getattr(second, "cp_loss", 0))

        unclear_margin = float(cfg.get("unclear_win_margin", 0.04))
        best_win = float(getattr(best_analysis, "win_probability", 0.5)) if best_analysis else 0.5
        near_equal = abs(best_win - 0.5) <= unclear_margin

        close_cp_loss = int(cfg.get("close_move_cp_loss", 30))
        close_moves = sum(1 for m in move_analyses if int(getattr(m, "cp_loss", 10**9)) <= close_cp_loss)

        forced_mate = False
        mate_in = getattr(best_analysis, "mate_in", None) if best_analysis else None
        if mate_in is not None:
            try:
                forced_mate = int(mate_in) > 0
            except (TypeError, ValueError):
                forced_mate = False

        only_move_cp_loss = int(cfg.get("only_move_cp_loss", 150))
        only_move_best_prob = float(cfg.get("only_move_best_prob", 0.55))
        only_move = bool(forced_mate)
        if second_cp_loss is not None and second_cp_loss >= only_move_cp_loss:
            only_move = True
        if best_prob >= only_move_best_prob and close_moves <= 2:
            only_move = True

        many_options_entropy = float(cfg.get("many_options_entropy", 0.82))
        many_options_best_prob = float(cfg.get("many_options_best_prob", 0.22))
        many_options = False
        if entropy_norm >= many_options_entropy and best_prob <= 0.35:
            many_options = True
        if best_prob <= many_options_best_prob and close_moves >= 3:
            many_options = True
        if second_cp_loss is not None and second_cp_loss <= int(cfg.get("many_options_second_cp_loss", 20)) and close_moves >= 3:
            many_options = True

        return {
            "best_prob": best_prob,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "effective_moves": effective_moves,
            "cp_gap": cp_gap,
            "second_cp_loss": second_cp_loss,
            "close_moves": close_moves,
            "near_equal": near_equal,
            "only_move": only_move,
            "forced_mate": forced_mate,
            "many_options": many_options,
        }

    def _decision_profile_line(
        self,
        board: chess.Board,
        position: Dict[str, Any],
        best_analysis: Optional[Any],
        decision_profile: Dict[str, Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if not bool(cfg.get("include_decision_profile", True)):
            return ""

        source = str(position.get("source") or "")
        prob = float(cfg.get("decision_profile_prob", 0.25))
        if source == "puzzle":
            prob *= float(cfg.get("puzzle_decision_profile_mult", 1.2))
        elif source == "game":
            prob *= float(cfg.get("game_decision_profile_mult", 0.8))

        if style == "quick":
            prob *= 0.25
        elif style == "concise":
            prob *= 0.6

        if rng.random() >= min(max(prob, 0.0), 1.0):
            return ""

        if decision_profile.get("forced_mate"):
            return rng.choice(
                [
                    "This looks like a forcing sequence—calculation matters.",
                    "This feels forcing; the king is under serious pressure.",
                    "The position looks tactical and forcing, so concrete lines matter most.",
                ]
            )

        if decision_profile.get("only_move"):
            if decision_profile.get("near_equal"):
                return rng.choice(
                    [
                        "This feels like a critical moment—there may be only one accurate move to stay afloat.",
                        "This position looks knife-edge; accuracy matters and alternatives may fail quickly.",
                        "It feels like an only-move situation: small inaccuracies can swing the evaluation.",
                    ]
                )
            return rng.choice(
                [
                    "The best move seems to stand out; most alternatives look clearly worse.",
                    "This feels close to an only-move position—precision matters.",
                    "It looks like accuracy is required here; the margin for error is small.",
                ]
            )

        if decision_profile.get("many_options"):
            if self._position_phase(board) == "opening":
                return rng.choice(
                    [
                        "Several moves look playable—pick the one that develops cleanly and keeps the king safe.",
                        "There are multiple reasonable choices; prioritize development and coordination.",
                    ]
                )
            if self._position_phase(board) == "endgame":
                return rng.choice(
                    [
                        "There are several reasonable options—choose the line that improves king activity and pawn structure.",
                        "Multiple moves seem close; focus on king activation and the clearest plan.",
                    ]
                )
            return rng.choice(
                [
                    "There are a few close options; choose the cleanest plan and avoid creating weaknesses.",
                    "Several moves look similar; I should prefer the move with the clearest follow-up.",
                    "This isn't a forced spot—multiple continuations look playable, so clarity matters.",
                ]
            )

        return rng.choice(
            [
                "There is some choice here, but I want the most concrete and least risky continuation.",
                "It looks like a position where a small nuance matters, so I'll lean on the most precise line.",
                "There are a couple ideas, but I want the move that improves coordination without loosening the king.",
            ]
        )

    def _context_hint_line(
        self,
        board: chess.Board,
        position: Dict[str, Any],
        analysis: Any,
        best_analysis: Optional[Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if not bool(cfg.get("include_context_hints", True)):
            return ""

        source = str(position.get("source") or "")
        base_prob = float(cfg.get("context_hint_prob", 0.18))
        if source == "puzzle":
            base_prob *= float(cfg.get("puzzle_context_hint_mult", 1.5))
        elif source == "game":
            base_prob *= float(cfg.get("game_context_hint_mult", 0.7))

        if style == "quick":
            base_prob *= 0.25
        elif style == "concise":
            base_prob *= 0.6

        if rng.random() >= min(max(base_prob, 0.0), 1.0):
            return ""

        themes = position.get("puzzle_themes") or []
        if isinstance(themes, str):
            themes = [t for t in themes.split() if t]
        themes_lc = {str(t).strip().lower() for t in themes if t}

        endgameish = any("endgame" in t for t in themes_lc) or self._position_phase(board) == "endgame"
        mateish = any("mate" in t for t in themes_lc) or (best_analysis is not None and getattr(best_analysis, "mate_in", None))
        loose_pieceish = any("hanging" in t or "trapped" in t for t in themes_lc)
        tacticalish = any(
            any(marker in t for marker in ("fork", "pin", "skewer", "deflection", "attraction", "interference", "discovered", "double", "backrank"))
            for t in themes_lc
        )

        if mateish:
            return rng.choice(
                [
                    "Hint: prioritize forcing moves and keep checking motifs in mind.",
                    "Hint: look for forcing checks and ways to restrict the king's escape squares.",
                    "Hint: start your search with checks and direct threats against the king.",
                ]
            )
        if endgameish:
            return rng.choice(
                [
                    "Hint: in endgames, king activity and pawn races often decide everything.",
                    "Hint: endgame focus—activate the king and keep an eye on passed pawns.",
                    "Hint: think about king activation and conversion technique here.",
                ]
            )
        if loose_pieceish:
            return rng.choice(
                [
                    "Hint: scan for loose pieces and tactical captures.",
                    "Hint: look for undefended pieces and simple tactics that win material.",
                    "Hint: check which pieces are hanging and whether a forcing capture exists.",
                ]
            )
        if tacticalish:
            return rng.choice(
                [
                    "Hint: start with checks, captures, and threats; tactics may dominate.",
                    "Hint: look for tactics like forks, pins, or overloaded defenders.",
                    "Hint: try to find the forcing sequence first (checks/captures/threats).",
                ]
            )
        return rng.choice(
            [
                "Hint: start with checks, captures, and threats, then consider improving moves.",
                "Hint: look for forcing moves first, then fall back to improving the worst-placed piece.",
                "Hint: identify the main threat and the cleanest way to address it.",
            ]
        )

    def _weighted_choice(self, weights: Dict[str, float], rng: random.Random) -> str:
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0:
            return next(iter(weights.keys()))
        pick = rng.random() * total
        running = 0.0
        for key, weight in weights.items():
            running += max(weight, 0.0)
            if pick <= running:
                return key
        return next(iter(weights.keys()))

    def _threat_scan_line(
        self,
        board: chess.Board,
        style: str,
        rng: random.Random,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        max_checks = int(cfg.get("threat_max_checks", 2))
        max_hanging = int(cfg.get("threat_max_hanging", 2))
        include_empty = bool(cfg.get("threat_scan_include_empty", False))
        side = "White" if board.turn == chess.WHITE else "Black"
        lines = []

        if board.is_check():
            lines.append(f"{side} is in check and must respond.")

        checks = [move for move in board.legal_moves if board.gives_check(move)]
        if checks:
            moves_text = self._format_moves(board, checks, notation, max_checks)
            extra = " and others" if len(checks) > max_checks else ""
            lines.append(f"Checks to consider: {moves_text}{extra}.")

        opponent_board = board.copy()
        opponent_board.turn = not board.turn
        opp_checks = [move for move in opponent_board.legal_moves if opponent_board.gives_check(move)]
        if opp_checks:
            moves_text = self._format_moves(opponent_board, opp_checks, notation, max_checks)
            lines.append(f"Opponent has checking ideas like {moves_text}.")

        hanging = self._find_hanging_pieces(board, max_hanging)
        if hanging:
            pieces = []
            for piece, square in hanging:
                color = "White" if piece.color == chess.WHITE else "Black"
                square_name = chess.square_name(square)
                piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
                pieces.append(f"{color} {piece_name} on {square_name}")
            lines.append(f"Hanging pieces: {', '.join(pieces)}.")

        if not lines:
            if include_empty:
                return rng.choice([
                    "No immediate tactics stand out.",
                    "No forcing tactics jump out.",
                ])
            return ""

        prefix = rng.choice([
            "Quick threat scan:",
            "Immediate tactics:",
            "Tactical scan:",
        ])
        return f"{prefix} " + " ".join(lines)

    def _plan_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if board.is_check():
            return rng.choice([
                "First priority is to get out of check cleanly.",
                "The immediate task is handling the check safely.",
            ])

        color = board.turn
        plan_options: List[str] = []

        king_sq = board.king(color)
        if king_sq in (chess.E1, chess.E8):
            if board.has_kingside_castling_rights(color):
                plan_options.append(rng.choice([
                    "Plan: castle kingside and consolidate.",
                    "I want to get the king safe with O-O.",
                ]))
            if board.has_queenside_castling_rights(color):
                plan_options.append(rng.choice([
                    "Plan: castle long and start activity on the kingside.",
                    "Queenside castling is an option to keep the initiative.",
                ]))

        passed_pawn = self._find_passed_pawn(board, color)
        if passed_pawn is not None:
            square_name = chess.square_name(passed_pawn)
            plan_options.append(rng.choice([
                f"The passed pawn on {square_name} is a long-term asset to push.",
                f"I should look to advance the passed pawn on {square_name}.",
            ]))

        open_files = self._open_files(board)
        rooks = board.pieces(chess.ROOK, color)
        for square in rooks:
            file_idx = chess.square_file(square)
            if file_idx in open_files:
                file_letter = "abcdefgh"[file_idx]
                plan_options.append(rng.choice([
                    f"Occupy the open {file_letter}-file with a rook.",
                    f"Rooks belong on the open {file_letter}-file.",
                ]))
                break

        if self._is_exposed_king(board, not color):
            plan_options.append(rng.choice([
                "The enemy king looks exposed, so an attack is natural.",
                "There are chances to go after the king with active pieces.",
            ]))

        if not plan_options:
            plan_options.append(rng.choice([
                "Plan: improve piece activity and keep options open.",
                "I want to coordinate pieces and avoid creating weaknesses.",
                "The plan is to finish development and centralize.",
            ]))

        return rng.choice(plan_options)

    def _opponent_perspective_line(
        self,
        board: chess.Board,
        style: str,
        rng: random.Random,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        opponent = not board.turn
        opp_board = board.copy()
        opp_board.turn = opponent
        checks = [move for move in opp_board.legal_moves if opp_board.gives_check(move)]
        if checks:
            moves_text = self._format_moves(opp_board, checks, notation, int(cfg.get("threat_max_checks", 2)))
            return rng.choice([
                f"From the opponent's view, checks like {moves_text} are the main forcing ideas.",
                f"Opponent may look for checks such as {moves_text}.",
            ])

        hanging = self._find_hanging_pieces_for_color(board, board.turn, 1)
        if hanging:
            piece, square = hanging[0]
            piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
            square_name = chess.square_name(square)
            return rng.choice([
                f"Opponent will try to pick off the {piece_name} on {square_name}.",
                f"The {piece_name} on {square_name} could be a target for the opponent.",
            ])

        return rng.choice([
            "From the opponent's side, the plan is likely to challenge the center.",
            "Opponent probably wants to simplify and relieve the pressure.",
            "Expect the opponent to activate pieces and look for counterplay.",
        ])

    def _format_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        notation: str,
        max_items: int,
    ) -> str:
        shown = moves[:max_items]
        labels = [self._move_notation(board, move, notation) for move in shown]
        return ", ".join(labels)

    def _move_notation(self, board: chess.Board, move: chess.Move, notation: str) -> str:
        if notation == "san":
            try:
                return board.san(move)
            except Exception:
                return move.uci()
        return move.uci()

    def _find_hanging_pieces(
        self,
        board: chess.Board,
        max_items: int,
    ) -> List[Tuple[chess.Piece, chess.Square]]:
        candidates: List[Tuple[int, int, chess.Square, chess.Piece]] = []
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
            attackers = board.attackers(not piece.color, square)
            if not attackers:
                continue
            defenders = board.attackers(piece.color, square)
            if defenders:
                continue
            value = PIECE_VALUES.get(piece.piece_type, 0)
            turn_bias = 1 if piece.color == board.turn else 0
            candidates.append((value, turn_bias, square, piece))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [(piece, square) for _, _, square, piece in candidates[:max_items]]

    def _find_hanging_pieces_for_color(
        self,
        board: chess.Board,
        color: bool,
        max_items: int,
    ) -> List[Tuple[chess.Piece, chess.Square]]:
        candidates: List[Tuple[int, chess.Square, chess.Piece]] = []
        for square, piece in board.piece_map().items():
            if piece.color != color or piece.piece_type == chess.KING:
                continue
            attackers = board.attackers(not color, square)
            if not attackers:
                continue
            defenders = board.attackers(color, square)
            if defenders:
                continue
            value = PIECE_VALUES.get(piece.piece_type, 0)
            candidates.append((value, square, piece))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [(piece, square) for _, square, piece in candidates[:max_items]]

    def _detect_candidate_motifs(
        self,
        board: chess.Board,
        move: chess.Move,
        cand: Any,
        cfg: Dict[str, Any],
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        motifs: List[str] = []
        max_motifs = int(cfg.get("max_motifs_per_candidate", 1))
        attacker_color = board.color_at(move.from_square)
        temp = board.copy()
        temp.push(move)
        if rng is None:
            rng = random.Random()

        mate_pattern = self._detect_mate_pattern(temp, move, attacker_color)
        if mate_pattern:
            motifs.append(mate_pattern)
            if len(motifs) >= max_motifs:
                return motifs

        underpromotion = self._detect_underpromotion(move)
        if underpromotion:
            motifs.append(underpromotion)
            if len(motifs) >= max_motifs:
                return motifs

        en_passant = self._detect_en_passant(board, move)
        if en_passant:
            motifs.append(en_passant)
            if len(motifs) >= max_motifs:
                return motifs

        double_check = self._detect_double_check(temp, attacker_color)
        if double_check:
            motifs.append(double_check)
            if len(motifs) >= max_motifs:
                return motifs
        fork = self._detect_fork(temp, move, attacker_color)
        if fork:
            motifs.append(fork)
            if len(motifs) >= max_motifs:
                return motifs
        pin = self._detect_pin(temp)
        if pin:
            motifs.append(pin)
            if len(motifs) >= max_motifs:
                return motifs
        skewer = self._detect_skewer(temp, attacker_color)
        if skewer:
            motifs.append(skewer)
            if len(motifs) >= max_motifs:
                return motifs
        discovered = self._detect_discovered_attack(board, move, attacker_color)
        if discovered:
            motifs.append(discovered)
            if len(motifs) >= max_motifs:
                return motifs
        else:
            clearance = self._detect_clearance(board, move, attacker_color)
            if clearance:
                motifs.append(clearance)
                if len(motifs) >= max_motifs:
                    return motifs
        deflection = self._detect_deflection(board, move, attacker_color)
        if deflection:
            motifs.append(deflection)
            if len(motifs) >= max_motifs:
                return motifs

        interference = self._detect_interference(board, move, attacker_color, after_board=temp, rng=rng)
        if interference:
            motifs.append(interference)
            if len(motifs) >= max_motifs:
                return motifs

        xray = self._detect_xray(temp, attacker_color)
        if xray:
            motifs.append(xray)
            if len(motifs) >= max_motifs:
                return motifs
        sacrifice = self._detect_sacrifice(board, move, attacker_color, after_board=temp)
        if sacrifice:
            motifs.append(sacrifice)
            if len(motifs) >= max_motifs:
                return motifs
        attraction = self._detect_attraction(board, move, attacker_color, after_board=temp)
        if attraction:
            motifs.append(attraction)
        trapped = self._detect_trapped_piece(temp, attacker_color)
        if trapped:
            motifs.append(trapped)
        overload = self._detect_overload(temp, attacker_color)
        if overload:
            motifs.append(overload)
        back_rank = self._detect_back_rank_weakness(temp, attacker_color)
        if back_rank:
            motifs.append(back_rank)
        stalemate = self._detect_stalemate_trick(temp, attacker_color)
        if stalemate:
            motifs.append(stalemate)
        perpetual = self._detect_perpetual_idea(board, move, cand, attacker_color, cfg, after_board=temp)
        if perpetual:
            motifs.append(perpetual)
        zwischenzug = self._detect_zwischenzug(board, move, attacker_color)
        if zwischenzug:
            motifs.append(zwischenzug)
        weak = self._detect_weak_square_move(board, move, attacker_color)
        if weak:
            motifs.append(weak)
        quiet_move = self._detect_quiet_move(board, move, attacker_color, cfg)
        if quiet_move:
            motifs.append(quiet_move)

        # Advanced coordination and positional motifs
        if cfg.get("include_coordination", True):
            battery = self._detect_battery_formation(temp, move, attacker_color)
            if battery:
                motifs.append(battery)
            outpost = self._detect_knight_outpost(temp, move, attacker_color)
            if outpost:
                motifs.append(outpost)

        # Tempo awareness
        if cfg.get("include_tempo", True):
            tempo = self._detect_tempo_move(board, move, attacker_color)
            if tempo:
                motifs.append(tempo)

        # Prophylactic thinking
        if cfg.get("include_prophylactic", True):
            prophylactic = self._detect_prophylactic_move(board, move, attacker_color)
            if prophylactic:
                motifs.append(prophylactic)

        # Aesthetic/emotional commentary
        if cfg.get("include_aesthetic", True):
            aesthetic = self._detect_aesthetic_move(board, move, cand, attacker_color, rng)
            if aesthetic:
                motifs.append(aesthetic)

        return motifs[:max_motifs]

    def _detect_fork(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        piece = board.piece_at(move.to_square)
        if piece is None or attacker_color is None:
            return None
        enemy_color = not piece.color
        attacked = []
        for square in board.attacks(move.to_square):
            target = board.piece_at(square)
            if not target or target.color != enemy_color:
                continue
            if target.piece_type == chess.KING:
                continue
            value = PIECE_VALUES.get(target.piece_type, 0)
            if value >= 3:
                attacked.append((value, PIECE_NAMES.get(target.piece_type, "piece")))
        if len(attacked) < 2:
            return None
        attacked.sort(key=lambda x: x[0], reverse=True)
        first = attacked[0][1]
        second = attacked[1][1]
        if first == second:
            return f"fork on two {self._plural_piece(first)}"
        return f"fork on the {first} and {second}"

    def _detect_pin(self, board: chess.Board) -> Optional[str]:
        opponent_color = board.turn
        pinned: List[Tuple[int, chess.Piece, chess.Square]] = []
        for square, piece in board.piece_map().items():
            if piece.color != opponent_color:
                continue
            if board.is_pinned(opponent_color, square):
                value = PIECE_VALUES.get(piece.piece_type, 0)
                pinned.append((value, piece, square))
        if not pinned:
            return None
        pinned.sort(key=lambda x: x[0], reverse=True)
        _, piece, square = pinned[0]
        name = PIECE_NAMES.get(piece.piece_type, "piece")
        return f"pin on the {name} at {chess.square_name(square)}"

    def _detect_double_check(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        king_sq = board.king(not attacker_color)
        if king_sq is None:
            return None
        if not board.is_check():
            return None
        attackers = board.attackers(attacker_color, king_sq)
        if len(attackers) >= 2:
            return "double check"
        return None

    def _detect_skewer(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        enemy_color = not attacker_color
        sliders = (
            list(board.pieces(chess.ROOK, attacker_color)) +
            list(board.pieces(chess.BISHOP, attacker_color)) +
            list(board.pieces(chess.QUEEN, attacker_color))
        )
        for slider_sq in sliders:
            for direction in self._slider_directions(board.piece_at(slider_sq)):
                current = slider_sq
                first_enemy = None
                while True:
                    current = self._step_square(current, direction)
                    if current is None:
                        break
                    piece = board.piece_at(current)
                    if piece is None:
                        continue
                    if piece.color == attacker_color:
                        break
                    if first_enemy is None:
                        first_enemy = piece
                        continue
                    second_enemy = piece
                    if first_enemy.piece_type == chess.KING:
                        return "skewer on the king"
                    if PIECE_VALUES.get(first_enemy.piece_type, 0) >= PIECE_VALUES.get(second_enemy.piece_type, 0):
                        first_name = PIECE_NAMES.get(first_enemy.piece_type, "piece")
                        second_name = PIECE_NAMES.get(second_enemy.piece_type, "piece")
                        return f"skewer on the {first_name} to the {second_name}"
                    break
        return None

    def _detect_clearance(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        temp = board.copy()
        temp.push(move)
        king_sq = temp.king(not attacker_color)
        if king_sq is None or not temp.is_check():
            return None
        attackers = temp.attackers(attacker_color, king_sq)
        if move.to_square in attackers:
            return None
        return "clearance for a discovered check"

    def _detect_deflection(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.is_capture(move):
            return None
        captured_square, captured_piece = self._captured_piece(board, move)
        if captured_piece is None or captured_piece.color == attacker_color:
            return None
        defender_color = captured_piece.color
        temp = board.copy()
        temp.push(move)
        targets: List[Tuple[int, str, chess.Square]] = []
        for target_square in board.attacks(captured_square):
            target_piece = board.piece_at(target_square)
            if not target_piece or target_piece.color != defender_color:
                continue
            if target_piece.piece_type == chess.KING:
                continue
            if temp.is_attacked_by(attacker_color, target_square) and not temp.is_attacked_by(defender_color, target_square):
                value = PIECE_VALUES.get(target_piece.piece_type, 0)
                targets.append((value, PIECE_NAMES.get(target_piece.piece_type, "piece"), target_square))
        if not targets:
            return None
        targets.sort(key=lambda x: x[0], reverse=True)
        name = targets[0][1]
        return f"deflection of the {name}"

    def _detect_xray(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        sliders = (
            list(board.pieces(chess.ROOK, attacker_color)) +
            list(board.pieces(chess.BISHOP, attacker_color)) +
            list(board.pieces(chess.QUEEN, attacker_color))
        )
        for slider_sq in sliders:
            piece = board.piece_at(slider_sq)
            for direction in self._slider_directions(piece):
                current = slider_sq
                blocker = None
                while True:
                    current = self._step_square(current, direction)
                    if current is None:
                        break
                    target = board.piece_at(current)
                    if target is None:
                        continue
                    if blocker is None:
                        blocker = target
                        continue
                    if target.color != attacker_color and target.piece_type in (chess.KING, chess.QUEEN):
                        name = PIECE_NAMES.get(target.piece_type, "piece")
                        return f"x-ray on the {name}"
                    break
        return None

    def _detect_attraction(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
        after_board: Optional[chess.Board] = None,
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.gives_check(move):
            return None
        temp = after_board
        if temp is None:
            temp = board.copy()
            temp.push(move)
        moved_piece = temp.piece_at(move.to_square)
        if moved_piece is None:
            return None
        if temp.is_attacked_by(not attacker_color, move.to_square) and not temp.is_attacked_by(attacker_color, move.to_square):
            return "decoy on the king"
        return None

    def _detect_trapped_piece(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        enemy_color = not attacker_color
        candidates: List[Tuple[int, str, chess.Square]] = []
        for square, piece in board.piece_map().items():
            if piece.color != enemy_color:
                continue
            if piece.piece_type in (chess.KING, chess.PAWN):
                continue
            if not board.is_attacked_by(attacker_color, square):
                continue
            scratch = board.copy(stack=False)
            any_moves = False
            safe = False
            for mv in board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]):
                any_moves = True
                scratch.push(mv)
                safe = not scratch.is_attacked_by(attacker_color, mv.to_square)
                scratch.pop()
                if safe:
                    break
            if not any_moves:
                continue
            if not safe:
                value = PIECE_VALUES.get(piece.piece_type, 0)
                name = PIECE_NAMES.get(piece.piece_type, "piece")
                candidates.append((value, name, square))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        name = candidates[0][1]
        return f"trapped {name}"

    @staticmethod
    def _captured_piece(
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[chess.Square, Optional[chess.Piece]]:
        if board.is_en_passant(move):
            offset = -8 if board.turn == chess.WHITE else 8
            capture_square = move.to_square + offset
            return capture_square, board.piece_at(capture_square)
        return move.to_square, board.piece_at(move.to_square)

    def _detect_discovered_attack(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        for slider_sq in (
            list(board.pieces(chess.ROOK, attacker_color)) +
            list(board.pieces(chess.BISHOP, attacker_color)) +
            list(board.pieces(chess.QUEEN, attacker_color))
        ):
            directions = self._slider_directions(board.piece_at(slider_sq))
            for direction in directions:
                current = slider_sq
                blocked = False
                found_mover = False
                while True:
                    current = self._step_square(current, direction)
                    if current is None:
                        break
                    if current == move.from_square:
                        found_mover = True
                        continue
                    piece = board.piece_at(current)
                    if piece is None:
                        continue
                    if not found_mover:
                        blocked = True
                        break
                    if piece.color != attacker_color:
                        return "discovered attack"
                    break
                if found_mover and not blocked:
                    continue
        return None

    def _detect_overload(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        defender_color = not attacker_color
        overloaded: List[Tuple[int, chess.Piece, chess.Square]] = []
        for square, piece in board.piece_map().items():
            if piece.color != defender_color:
                continue
            defended_targets = 0
            for target_square in board.attacks(square):
                target_piece = board.piece_at(target_square)
                if not target_piece or target_piece.color != defender_color:
                    continue
                if board.attackers(attacker_color, target_square):
                    defended_targets += 1
            if defended_targets >= 2:
                value = PIECE_VALUES.get(piece.piece_type, 0)
                overloaded.append((value, piece, square))
        if not overloaded:
            return None
        overloaded.sort(key=lambda x: x[0], reverse=True)
        _, piece, square = overloaded[0]
        name = PIECE_NAMES.get(piece.piece_type, "piece")
        return f"overloaded defender on the {name} at {chess.square_name(square)}"

    def _detect_back_rank_weakness(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        defender_color = not attacker_color
        king_sq = board.king(defender_color)
        if king_sq is None:
            return None
        back_rank = 0 if defender_color == chess.WHITE else 7
        if chess.square_rank(king_sq) != back_rank:
            return None
        escape = False
        for square in board.attacks(king_sq):
            if board.piece_at(square) is not None:
                continue
            if board.is_attacked_by(attacker_color, square):
                continue
            escape = True
            break
        if not escape:
            return "back rank weakness"
        return None

    def _detect_stalemate_trick(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if board.is_check():
            return None
        if any(board.legal_moves):
            return None
        return "stalemate trick"

    def _detect_perpetual_idea(
        self,
        board: chess.Board,
        move: chess.Move,
        cand: Any,
        attacker_color: Optional[bool],
        cfg: Dict[str, Any],
        after_board: Optional[chess.Board] = None,
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.gives_check(move):
            return None
        pv_uci = list(getattr(cand, "pv_uci", None) or [])
        if not pv_uci:
            return None
        max_plies = int(cfg.get("perpetual_max_plies", 6))
        temp = after_board.copy(stack=False) if after_board is not None else board.copy()
        if after_board is None:
            temp.push(move)
        checks = 1
        for uci in pv_uci[1:max_plies + 1]:
            try:
                next_move = chess.Move.from_uci(uci)
            except Exception:
                break
            if next_move not in temp.legal_moves:
                break
            if temp.turn == attacker_color and temp.gives_check(next_move):
                checks += 1
            temp.push(next_move)
        if checks >= 2:
            return "perpetual check idea"
        return None

    def _detect_sacrifice(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
        after_board: Optional[chess.Board] = None,
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not (board.is_capture(move) or board.gives_check(move)):
            return None
        moved = board.piece_at(move.from_square)
        if moved is None or moved.color != attacker_color:
            return None
        moved_value = PIECE_VALUES.get(moved.piece_type, 0)
        temp = after_board
        if temp is None:
            temp = board.copy()
            temp.push(move)
        piece = temp.piece_at(move.to_square)
        if piece is None or piece.color != attacker_color:
            return None
        if not temp.is_attacked_by(not attacker_color, move.to_square):
            return None
        if temp.is_attacked_by(attacker_color, move.to_square):
            return None
        if moved_value < 3:
            return None

        pre_hanging = board.is_attacked_by(not attacker_color, move.from_square) and not board.is_attacked_by(
            attacker_color, move.from_square
        )
        if pre_hanging and board.is_capture(move):
            _, captured_piece = self._captured_piece(board, move)
            if captured_piece and captured_piece.color != attacker_color:
                captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
                if captured_value >= 3 and captured_value >= moved_value - 1:
                    return "desperado"

        return "sacrifice"

    def _detect_underpromotion(self, move: chess.Move) -> Optional[str]:
        if not move.promotion:
            return None
        if move.promotion == chess.QUEEN:
            return None
        name = PIECE_NAMES.get(move.promotion, "piece")
        return f"underpromotion to a {name}"

    def _detect_en_passant(self, board: chess.Board, move: chess.Move) -> Optional[str]:
        try:
            if board.is_en_passant(move):
                return "en passant tactic"
        except Exception:
            return None
        return None

    def _detect_mate_pattern(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.is_checkmate():
            return None

        if move.promotion and move.promotion != chess.QUEEN:
            promo = PIECE_NAMES.get(move.promotion, "piece")
            return f"underpromotion mate ({promo})"

        enemy_color = not attacker_color
        king_sq = board.king(enemy_color)
        moved_piece = board.piece_at(move.to_square)

        if king_sq is not None and moved_piece is not None:
            if moved_piece.piece_type == chess.KNIGHT:
                neighbors = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
                occupied = 0
                empty = 0
                for sq in neighbors:
                    piece = board.piece_at(sq)
                    if piece is None:
                        empty += 1
                    elif piece.color == enemy_color:
                        occupied += 1
                if occupied >= 3 and empty <= 1:
                    return "smothered mate"

            if moved_piece.piece_type in (chess.ROOK, chess.QUEEN):
                back_rank = 0 if enemy_color == chess.WHITE else 7
                if chess.square_rank(king_sq) == back_rank:
                    shield_rank = 1 if enemy_color == chess.WHITE else 6
                    pawn_shield = 0
                    for df in (-1, 0, 1):
                        file_idx = chess.square_file(king_sq) + df
                        if file_idx < 0 or file_idx > 7:
                            continue
                        sq = chess.square(file_idx, shield_rank)
                        piece = board.piece_at(sq)
                        if piece and piece.color == enemy_color and piece.piece_type == chess.PAWN:
                            pawn_shield += 1
                    if pawn_shield >= 2:
                        return "back rank mate"

        return "checkmate"

    def _detect_interference(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
        *,
        after_board: Optional[chess.Board] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if board.is_capture(move) or move.promotion:
            return None
        if after_board is None:
            after_board = board.copy()
            after_board.push(move)

        defender_color = not attacker_color
        blocker_sq = move.to_square

        # Look for a newly-hanging enemy piece where a *slider defender* was cut off by our interposition.
        for target_sq, target_piece in after_board.piece_map().items():
            if target_piece.color != defender_color or target_piece.piece_type == chess.KING:
                continue
            if not after_board.is_attacked_by(attacker_color, target_sq):
                continue
            if after_board.is_attacked_by(defender_color, target_sq):
                continue
            if not board.is_attacked_by(defender_color, target_sq):
                continue

            before_def = set(board.attackers(defender_color, target_sq))
            after_def = set(after_board.attackers(defender_color, target_sq))
            removed = before_def - after_def
            if not removed:
                continue

            for defender_sq in removed:
                defender_piece = board.piece_at(defender_sq)
                if defender_piece is None or defender_piece.color != defender_color:
                    continue
                if defender_piece.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
                    continue
                between = chess.BB_BETWEEN[defender_sq][target_sq]
                if not between:
                    continue
                if not (between & chess.BB_SQUARES[blocker_sq]):
                    continue

                target_name = PIECE_NAMES.get(target_piece.piece_type, "piece")
                target_square = chess.square_name(target_sq)
                if rng is not None and rng.random() < 0.5:
                    defender_name = PIECE_NAMES.get(defender_piece.piece_type, "piece")
                    return f"interference—blocks the {defender_name}'s defense of {target_name} on {target_square}"
                return f"interference—cuts defense of {target_name} on {target_square}"

        return None

    def _detect_zwischenzug(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not (board.is_capture(move) or board.gives_check(move)):
            return None
        hanging = self._find_hanging_pieces_for_color(board, attacker_color, 1)
        if not hanging:
            return None
        return "zwischenzug"

    def _detect_quiet_move(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not cfg.get("include_quiet_move_motif", True):
            return None
        if board.is_capture(move) or board.gives_check(move) or move.promotion:
            return None
        if board.is_castling(move):
            return None
        return "quiet move"

    # =========================================================================
    # NEW ADVANCED DETECTION METHODS
    # =========================================================================

    def _detect_battery_formation(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        """Detect if move creates a battery (Q+B diagonal or Q+R/R+R on file)."""
        if attacker_color is None:
            return None

        to_sq = move.to_square
        piece = board.piece_at(to_sq)
        if piece is None:
            return None

        to_file = chess.square_file(to_sq)

        # Check for rook battery on same file
        if piece.piece_type == chess.ROOK:
            other_rooks = list(board.pieces(chess.ROOK, attacker_color))
            for rook_sq in other_rooks:
                if rook_sq != to_sq and chess.square_file(rook_sq) == to_file:
                    return "doubled rooks on the file"

        # Check for queen + rook battery on file
        if piece.piece_type in (chess.QUEEN, chess.ROOK):
            queens = list(board.pieces(chess.QUEEN, attacker_color))
            rooks = list(board.pieces(chess.ROOK, attacker_color))
            for q_sq in queens:
                if q_sq != to_sq and chess.square_file(q_sq) == to_file:
                    return "queen and rook battery"
            for r_sq in rooks:
                if r_sq != to_sq and piece.piece_type == chess.QUEEN:
                    if chess.square_file(r_sq) == to_file:
                        return "queen and rook battery"

        # Check for queen + bishop battery on diagonal
        if piece.piece_type in (chess.QUEEN, chess.BISHOP):
            def same_diagonal(sq1: chess.Square, sq2: chess.Square) -> bool:
                return abs(chess.square_file(sq1) - chess.square_file(sq2)) == \
                       abs(chess.square_rank(sq1) - chess.square_rank(sq2))

            queens = list(board.pieces(chess.QUEEN, attacker_color))
            bishops = list(board.pieces(chess.BISHOP, attacker_color))

            for q_sq in queens:
                if q_sq != to_sq and same_diagonal(q_sq, to_sq):
                    return "queen and bishop battery on the diagonal"
            for b_sq in bishops:
                if b_sq != to_sq and piece.piece_type == chess.QUEEN:
                    if same_diagonal(b_sq, to_sq):
                        return "queen and bishop battery"

        return None

    def _detect_knight_outpost(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        """Detect if move places knight on a strong outpost."""
        if attacker_color is None:
            return None

        piece = board.piece_at(move.to_square)
        if piece is None or piece.piece_type != chess.KNIGHT:
            return None

        to_sq = move.to_square
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)

        # Check if in enemy territory (ranks 4-6 for white, 3-5 for black)
        if attacker_color == chess.WHITE:
            if to_rank < 3:  # Not advanced enough
                return None
        else:
            if to_rank > 4:  # Not advanced enough
                return None

        # Check if defended by own pawn
        if not self._pawn_defended(board, attacker_color, to_sq):
            return None

        # Check if can't be attacked by enemy pawns
        enemy_color = not attacker_color
        enemy_pawns = board.pieces(chess.PAWN, enemy_color)

        # Check adjacent files for enemy pawns that could attack
        for adj_file in [to_file - 1, to_file + 1]:
            if 0 <= adj_file <= 7:
                for pawn_sq in enemy_pawns:
                    pawn_file = chess.square_file(pawn_sq)
                    pawn_rank = chess.square_rank(pawn_sq)
                    if pawn_file == adj_file:
                        # Enemy pawn could potentially attack
                        if attacker_color == chess.WHITE and pawn_rank > to_rank:
                            return None  # Pawn can advance and attack
                        if attacker_color == chess.BLACK and pawn_rank < to_rank:
                            return None

        return "strong knight outpost"

    def _detect_bishop_pair_advantage(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        """Detect if side has bishop pair advantage."""
        if attacker_color is None:
            return None

        own_bishops = len(list(board.pieces(chess.BISHOP, attacker_color)))
        opp_bishops = len(list(board.pieces(chess.BISHOP, not attacker_color)))

        if own_bishops == 2 and opp_bishops < 2:
            return "bishop pair advantage"
        return None

    def _detect_prophylactic_move(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        """Detect if move prevents a significant opponent threat."""
        if attacker_color is None:
            return None

        # Count opponent threats BEFORE move
        threats_before = self._count_significant_threats(board, not attacker_color)

        # Apply move and count threats AFTER
        test_board = board.copy()
        test_board.push(move)
        threats_after = self._count_significant_threats(test_board, not attacker_color)

        # If threats significantly reduced, it's prophylactic
        if threats_before > 0 and threats_after < threats_before:
            reduction = threats_before - threats_after
            if reduction >= 2:
                return "prevents multiple opponent threats"
            elif reduction == 1:
                return "prophylactic - stops opponent's plan"

        return None

    def _count_significant_threats(
        self,
        board: chess.Board,
        color: bool,
    ) -> int:
        """Count significant threats (checks, attacks on hanging pieces)."""
        count = 0

        # Count checking moves
        test_board = board.copy()
        test_board.turn = color
        for m in test_board.legal_moves:
            if test_board.gives_check(m):
                count += 1
                break  # One check is enough to count

        # Count attacks on undefended pieces
        opponent = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == opponent:
                if board.is_attacked_by(color, sq):
                    if not board.is_attacked_by(opponent, sq):
                        count += 1  # Hanging piece under attack

        return count

    def _detect_tempo_move(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        """Detect if move gains tempo by developing while attacking."""
        if attacker_color is None:
            return None

        piece = board.piece_at(move.from_square)
        if piece is None:
            return None

        # Check if it's a developing move (from back ranks)
        from_rank = chess.square_rank(move.from_square)
        is_developing = False
        if attacker_color == chess.WHITE:
            is_developing = from_rank <= 1  # From ranks 1-2
        else:
            is_developing = from_rank >= 6  # From ranks 7-8

        if not is_developing:
            return None

        # Check if gives check
        if board.gives_check(move):
            return "develops with check"

        # Check if move creates a threat after being played
        test_board = board.copy()
        test_board.push(move)

        # Check if attacks a piece
        to_sq = move.to_square
        attacks = test_board.attacks(to_sq)
        for attacked_sq in attacks:
            target = test_board.piece_at(attacked_sq)
            if target and target.color != attacker_color:
                if target.piece_type in (chess.QUEEN, chess.ROOK):
                    return "develops while attacking heavy piece"
                elif target.piece_type in (chess.KNIGHT, chess.BISHOP):
                    return "develops with tempo"

        return None

    def _detect_aesthetic_move(
        self,
        board: chess.Board,
        move: chess.Move,
        cand: Any,
        attacker_color: Optional[bool],
        rng: random.Random,
    ) -> Optional[str]:
        """Detect aesthetically notable moves (sacrifices, surprising quiet moves)."""
        if attacker_color is None:
            return None

        # Check for sacrifice (losing material but best move)
        if board.is_capture(move):
            piece = board.piece_at(move.from_square)
            captured = board.piece_at(move.to_square)
            if piece and captured:
                piece_val = PIECE_VALUES.get(piece.piece_type, 0)
                captured_val = PIECE_VALUES.get(captured.piece_type, 0)
                if piece_val > captured_val + 1:  # Sacrificing more than captured
                    win_prob = getattr(cand, "win_probability", 0.5)
                    if win_prob > 0.55:  # But still winning
                        return rng.choice(PHRASE_LIBRARY.get("beautiful_sacrifice", ["brilliant sacrifice"]))

        # Check for quiet move that's surprisingly best
        if not board.is_capture(move) and not board.gives_check(move):
            # If there were captures available but quiet move is best
            has_captures = any(board.is_capture(m) for m in board.legal_moves)
            if has_captures:
                win_prob = getattr(cand, "win_probability", 0.5)
                if win_prob > 0.55:
                    return rng.choice(PHRASE_LIBRARY.get("surprising_quiet_move", ["surprisingly quiet"]))

        return None

    @staticmethod
    def _slider_directions(piece: Optional[chess.Piece]) -> List[Tuple[int, int]]:
        if piece is None:
            return []
        if piece.piece_type == chess.ROOK:
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if piece.piece_type == chess.BISHOP:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if piece.piece_type == chess.QUEEN:
            return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        return []

    @staticmethod
    def _step_square(square: chess.Square, direction: Tuple[int, int]) -> Optional[chess.Square]:
        file_idx = chess.square_file(square) + direction[0]
        rank_idx = chess.square_rank(square) + direction[1]
        if file_idx < 0 or file_idx > 7 or rank_idx < 0 or rank_idx > 7:
            return None
        return chess.square(file_idx, rank_idx)

    def _detect_weak_square_move(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        weak_squares = self._find_weak_squares(board, attacker_color)
        if move.to_square in weak_squares:
            return f"weak square on {chess.square_name(move.to_square)}"
        return None

    def _find_weak_squares(
        self,
        board: chess.Board,
        attacker_color: bool,
    ) -> List[chess.Square]:
        opponent_color = not attacker_color
        weak: List[chess.Square] = []
        for square in WEAK_SQUARE_CANDIDATES:
            rank = chess.square_rank(square)
            if attacker_color == chess.WHITE and rank < 4:
                continue
            if attacker_color == chess.BLACK and rank > 3:
                continue
            if self._pawn_defended(board, opponent_color, square):
                continue
            if not board.attackers(attacker_color, square):
                continue
            weak.append(square)
        return weak

    @staticmethod
    def _pawn_defended(board: chess.Board, color: bool, square: chess.Square) -> bool:
        for attacker in board.attackers(color, square):
            piece = board.piece_at(attacker)
            if piece and piece.piece_type == chess.PAWN:
                return True
        return False

    @staticmethod
    def _open_files(board: chess.Board) -> List[int]:
        pawn_files = set()
        for square in board.pieces(chess.PAWN, chess.WHITE):
            pawn_files.add(chess.square_file(square))
        for square in board.pieces(chess.PAWN, chess.BLACK):
            pawn_files.add(chess.square_file(square))
        return [idx for idx in range(8) if idx not in pawn_files]

    def _half_open_files(
        self,
        board: chess.Board,
        color: bool,
    ) -> List[int]:
        friendly_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
        enemy_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)}
        return [idx for idx in range(8) if idx not in friendly_files and idx in enemy_files]

    @staticmethod
    def _pawn_file_counts(
        board: chess.Board,
        color: bool,
    ) -> Dict[int, List[chess.Square]]:
        files: Dict[int, List[chess.Square]] = {}
        for square in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(square)
            files.setdefault(file_idx, []).append(square)
        return files

    def _find_isolated_pawn(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        pawn_files = self._pawn_file_counts(board, color)
        if not pawn_files:
            return None
        candidates = []
        for file_idx, squares in pawn_files.items():
            if (file_idx - 1) in pawn_files or (file_idx + 1) in pawn_files:
                continue
            for square in squares:
                candidates.append(square)
        if not candidates:
            return None
        if color == chess.WHITE:
            return max(candidates, key=chess.square_rank)
        return min(candidates, key=chess.square_rank)

    def _find_doubled_pawn_file(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[int]:
        pawn_files = self._pawn_file_counts(board, color)
        doubled = [file_idx for file_idx, squares in pawn_files.items() if len(squares) >= 2]
        if not doubled:
            return None
        return sorted(doubled)[0]

    def _find_backward_pawn(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        pawn_files = self._pawn_file_counts(board, color)
        if not pawn_files:
            return None
        direction = 1 if color == chess.WHITE else -1
        candidates = []
        for file_idx, squares in pawn_files.items():
            for square in squares:
                rank = chess.square_rank(square)
                adjacent_pawns = []
                for adj_file in (file_idx - 1, file_idx + 1):
                    if adj_file in pawn_files:
                        adjacent_pawns.extend(pawn_files[adj_file])
                if not adjacent_pawns:
                    continue
                if color == chess.WHITE:
                    if not all(chess.square_rank(p) > rank for p in adjacent_pawns):
                        continue
                else:
                    if not all(chess.square_rank(p) < rank for p in adjacent_pawns):
                        continue
                next_rank = rank + direction
                if next_rank < 0 or next_rank > 7:
                    continue
                forward_sq = chess.square(file_idx, next_rank)
                if board.piece_at(forward_sq) is not None:
                    continue
                if board.is_attacked_by(not color, forward_sq) and not board.is_attacked_by(color, forward_sq):
                    candidates.append(square)
        if not candidates:
            return None
        if color == chess.WHITE:
            return min(candidates, key=chess.square_rank)
        return max(candidates, key=chess.square_rank)

    @staticmethod
    def _pawn_majority(
        board: chess.Board,
        color: bool,
    ) -> Optional[str]:
        pawns = list(board.pieces(chess.PAWN, color))
        if not pawns:
            return None
        queenside = sum(1 for sq in pawns if chess.square_file(sq) <= 3)
        kingside = sum(1 for sq in pawns if chess.square_file(sq) >= 4)
        if abs(queenside - kingside) < 2:
            return None
        if queenside > kingside:
            return "queenside pawn majority"
        return "kingside pawn majority"

    def _find_passed_pawn(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        pawns = list(board.pieces(chess.PAWN, color))
        opponent_pawns = list(board.pieces(chess.PAWN, not color))
        for pawn_sq in pawns:
            file_idx = chess.square_file(pawn_sq)
            rank_idx = chess.square_rank(pawn_sq)
            passed = True
            for opp_sq in opponent_pawns:
                opp_file = chess.square_file(opp_sq)
                if abs(opp_file - file_idx) > 1:
                    continue
                opp_rank = chess.square_rank(opp_sq)
                if color == chess.WHITE and opp_rank > rank_idx:
                    passed = False
                    break
                if color == chess.BLACK and opp_rank < rank_idx:
                    passed = False
                    break
            if passed:
                return pawn_sq
        return None

    def _find_outpost(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        minors = list(board.pieces(chess.KNIGHT, color)) + list(board.pieces(chess.BISHOP, color))
        for square in minors:
            rank = chess.square_rank(square)
            if color == chess.WHITE and rank < 4:
                continue
            if color == chess.BLACK and rank > 3:
                continue
            if not self._pawn_defended(board, color, square):
                continue
            if self._pawn_defended(board, not color, square):
                continue
            return square
        return None

    def _rook_on_seventh(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[str]:
        target_rank = 6 if color == chess.WHITE else 1
        for square in board.pieces(chess.ROOK, color):
            if chess.square_rank(square) == target_rank:
                return "rook on the 7th rank" if color == chess.WHITE else "rook on the 2nd rank"
        return None

    def _is_exposed_king(
        self,
        board: chess.Board,
        color: bool,
    ) -> bool:
        king_sq = board.king(color)
        if king_sq is None:
            return False
        file_idx = chess.square_file(king_sq)
        rank_idx = chess.square_rank(king_sq)
        shield_rank = rank_idx + 1 if color == chess.WHITE else rank_idx - 1
        if shield_rank < 0 or shield_rank > 7:
            return False
        pawn_count = 0
        for delta in (-1, 0, 1):
            file_check = file_idx + delta
            if file_check < 0 or file_check > 7:
                continue
            square = chess.square(file_check, shield_rank)
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type == chess.PAWN:
                pawn_count += 1
        return pawn_count < 2

    @staticmethod
    def _plural_piece(name: str) -> str:
        if name.endswith("s"):
            return name
        if name.endswith("y"):
            return name[:-1] + "ies"
        return name + "s"

    def _motif_phrase(self, motifs: List[str], rng: random.Random) -> str:
        if not motifs:
            return ""
        prefix = rng.choice([
            "Motif:",
            "Tactical motif:",
            "Pattern:",
            "Notice:",
        ])
        return f"{prefix} {'; '.join(motifs)}."

    def _detect_trap_for_candidate(
        self,
        cand: Any,
        analysis: Any,
        cfg: Dict[str, Any],
    ) -> Optional[Tuple[str, str, str]]:
        shallow_cps = getattr(analysis, "shallow_move_cps", None) or {}
        shallow_win = getattr(analysis, "shallow_move_win_probs", None) or {}
        if not shallow_cps and not shallow_win:
            return None

        shallow_cp = shallow_cps.get(cand.uci)
        shallow_wp = shallow_win.get(cand.uci)
        cp_values = [v for v in shallow_cps.values() if v is not None]
        wp_values = [v for v in shallow_win.values() if v is not None]
        best_shallow_cp = max(cp_values) if cp_values else None
        best_shallow_wp = max(wp_values) if wp_values else None

        trap_min_cp_swing = int(cfg.get("trap_min_cp_swing", 80))
        trap_min_wp_swing = float(cfg.get("trap_min_win_prob_swing", 0.12))
        trap_shallow_good_cp = int(cfg.get("trap_shallow_good_cp", 30))
        trap_shallow_bad_cp = int(cfg.get("trap_shallow_bad_cp", 120))
        trap_shallow_good_wp = float(cfg.get("trap_shallow_good_win_prob", 0.06))
        trap_shallow_bad_wp = float(cfg.get("trap_shallow_bad_win_prob", 0.18))

        deep_cp = getattr(cand, "centipawn", None)
        deep_wp = getattr(cand, "win_probability", None)

        if shallow_cp is not None and deep_cp is not None and best_shallow_cp is not None:
            cp_swing = abs(deep_cp - shallow_cp)
            shallow_good = shallow_cp >= best_shallow_cp - trap_shallow_good_cp
            shallow_bad = shallow_cp <= best_shallow_cp - trap_shallow_bad_cp
            deep_bad = deep_cp <= best_shallow_cp - trap_shallow_bad_cp
            deep_good = deep_cp >= best_shallow_cp - trap_shallow_good_cp

            if cp_swing >= trap_min_cp_swing and shallow_good and deep_bad:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "trap",
                    "looks strong at first glance but deeper analysis shows a problem",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )
            if cp_swing >= trap_min_cp_swing and shallow_bad and deep_good:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "resource",
                    "looks dubious at first glance but deeper analysis improves it",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )

        if shallow_wp is not None and deep_wp is not None and best_shallow_wp is not None:
            wp_swing = abs(deep_wp - shallow_wp)
            shallow_good = shallow_wp >= best_shallow_wp - trap_shallow_good_wp
            shallow_bad = shallow_wp <= best_shallow_wp - trap_shallow_bad_wp
            deep_bad = deep_wp <= best_shallow_wp - trap_shallow_bad_wp
            deep_good = deep_wp >= best_shallow_wp - trap_shallow_good_wp

            if wp_swing >= trap_min_wp_swing and shallow_good and deep_bad:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "trap",
                    "looks tempting at shallow depth but deeper analysis refutes it",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )
            if wp_swing >= trap_min_wp_swing and shallow_bad and deep_good:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "resource",
                    "looks quiet at shallow depth but hides a stronger idea",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )

        return None

    def _trap_phrase(
        self,
        trap: Optional[Tuple[str, str, str]],
        rng: random.Random,
        board: chess.Board,
        cand: Any,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        if not trap:
            return ""
        kind, detail, confidence = trap
        if confidence != "confirmed":
            lead = rng.choice([
                "Possible trap:",
                "Likely trap:",
                "Tricky trap:",
            ]) if kind == "trap" else rng.choice([
                "Possible resource:",
                "Hidden resource:",
                "Looks better than it seems:",
            ])
            line = f"{lead} {detail}."
            refutation = self._trap_refutation_line(board, cand, notation, cfg)
            if refutation:
                return f"{line} {refutation}"
            return line
        if kind == "trap":
            lead = rng.choice([
                "Confirmed trap:",
                "Trap alert:",
                "Trap warning:",
                "Trap caution:",
            ])
        else:
            lead = rng.choice([
                "Confirmed resource:",
                "Hidden resource:",
                "Quiet resource:",
                "Surprise:",
            ])
        line = f"{lead} {detail}."
        refutation = self._trap_refutation_line(board, cand, notation, cfg)
        if refutation:
            return f"{line} {refutation}"
        return line

    def _confirm_trap(
        self,
        cand: Any,
        analysis: Any,
        cfg: Dict[str, Any],
        kind: str,
        detail: str,
        deep_cp: Optional[int] = None,
        deep_wp: Optional[float] = None,
    ) -> Tuple[str, str, str]:
        confirm_cps = getattr(analysis, "confirm_move_cps", None) or {}
        confirm_wps = getattr(analysis, "confirm_move_win_probs", None) or {}
        confirm_cp = confirm_cps.get(cand.uci)
        confirm_wp = confirm_wps.get(cand.uci)

        confidence = "uncertain"
        if confirm_cp is not None and deep_cp is not None:
            tol = int(cfg.get("trap_confirm_cp_tolerance", 30))
            confidence = "confirmed" if abs(confirm_cp - deep_cp) <= tol else "uncertain"
        elif confirm_wp is not None and deep_wp is not None:
            tol = float(cfg.get("trap_confirm_win_prob_tolerance", 0.05))
            confidence = "confirmed" if abs(confirm_wp - deep_wp) <= tol else "uncertain"

        return (kind, detail, confidence)

    def _apply_guardrails(
        self,
        text: str,
        tablebase: Optional[TablebaseInfo],
        cfg: Dict[str, Any],
    ) -> str:
        if not text:
            return text
        if not cfg.get("guardrail_enabled", True):
            return text

        if tablebase and tablebase.result == "drawing":
            text = self._replace_word(text, "winning", "drawn")
            text = self._replace_word(text, "losing", "drawn")
            text = self._replace_word(text, "win", "draw")
            text = self._replace_word(text, "loss", "draw")
        return text

    @staticmethod
    def _replace_word(text: str, target: str, replacement: str) -> str:
        pattern = re.compile(rf"\\b{re.escape(target)}\\b", flags=re.IGNORECASE)

        def repl(match: re.Match[str]) -> str:
            word = match.group(0)
            if word.isupper():
                return replacement.upper()
            if word[:1].isupper():
                return replacement.capitalize()
            return replacement

        return pattern.sub(repl, text)

    def _trap_refutation_line(
        self,
        board: chess.Board,
        cand: Any,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        max_len = int(cfg.get("trap_refutation_max_len", 4))
        if max_len <= 0:
            return ""
        pv_uci = list(getattr(cand, "pv_uci", None) or [])
        if not pv_uci:
            return ""

        start_board = board
        if pv_uci and pv_uci[0] == getattr(cand, "uci", ""):
            pv_uci = pv_uci[1:]
            try:
                start_board = board.copy()
                start_board.push(chess.Move.from_uci(getattr(cand, "uci", "")))
            except Exception:
                start_board = board

        if not pv_uci:
            return ""

        pruned_moves, _ = self._prune_pv_moves(start_board, pv_uci[:max_len], cfg)
        line = self._format_pv_line(start_board, pruned_moves, notation)
        if not line:
            return ""
        return f"Line: {line}."

    def _format_pv_line(
        self,
        board: chess.Board,
        pv_uci: List[str],
        notation: str,
    ) -> str:
        if not pv_uci:
            return ""
        if notation != "san":
            return " ".join(pv_uci)

        temp = board.copy()
        san_moves: List[str] = []
        for uci in pv_uci:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in temp.legal_moves:
                break
            try:
                san_moves.append(temp.san(move))
            except Exception:
                san_moves.append(uci)
            temp.push(move)
        return " ".join(san_moves)

    def _apply_budget(
        self,
        sections: List[Tuple[str, str]],
        exploration_paragraphs: List[str],
        cfg: Dict[str, Any],
    ) -> str:
        max_tokens = cfg.get("max_trace_tokens")
        if not max_tokens:
            return self._join_sections(sections)

        def build_text() -> str:
            return self._join_sections(sections)

        text = build_text()
        if self._count_tokens(text) <= max_tokens:
            return text

        # First, trim exploration paragraphs.
        if exploration_paragraphs:
            while exploration_paragraphs and self._count_tokens(text) > max_tokens:
                exploration_paragraphs.pop()
                for idx, (name, _) in enumerate(sections):
                    if name == "exploration":
                        sections[idx] = (
                            name,
                            "\n\n".join(exploration_paragraphs),
                        )
                        break
                text = build_text()
            if self._count_tokens(text) <= max_tokens:
                return text

        # Drop optional sections in priority order.
        drop_order = [
            "avoid_blunder",
            "opening_plan",
            "endgame_technique",
            "decision_profile",
            "context_hint",
            "comparison",
            "dead_end",
            "reconsideration",
            "fen_walkthrough",
            "static_eval",
            "exploration",
            "candidates",
            "threat_scan",
            "opponent",
            "plan",
            "assessment",
            "orientation",
        ]
        for drop in drop_order:
            if self._count_tokens(text) <= max_tokens:
                break
            sections = [
                (name, body if name != drop else "")
                for name, body in sections
            ]
            text = build_text()

        if self._count_tokens(text) <= max_tokens:
            return text

        # Final fallback: return conclusion only.
        conclusion = ""
        for name, body in sections:
            if name == "conclusion":
                conclusion = body
                break
        return conclusion

    def _join_sections(self, sections: List[Tuple[str, str]]) -> str:
        return "\n\n".join(body for _, body in sections if body)

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        approx_words = text.count(" ") + text.count("\n") + text.count("\t") + 1
        return max(1, max(approx_words, len(text) // 4))

    @staticmethod
    def _repo_root() -> Path:
        cursor = Path(__file__).resolve()
        for parent in [cursor.parent, *cursor.parents]:
            if (parent / ".git").exists() or (parent / "README.md").exists():
                return parent
        return cursor.parent.parent

    @classmethod
    def _default_opening_paths(cls) -> List[Path]:
        root = cls._repo_root()
        candidates = [
            root / "data" / "openings",
            root / "data" / "chess-openings",
            root / "data" / "chess_openings",
        ]
        paths: List[Path] = []
        for folder in candidates:
            if folder.exists():
                paths.extend(sorted(folder.glob("*.tsv")))
        return paths

    @classmethod
    def _default_tablebase_paths(cls) -> List[Path]:
        root = cls._repo_root()
        candidates = [
            root / "data" / "syzygy",
            root / "data" / "tablebases",
            root / "data" / "tablebase",
        ]
        return [folder for folder in candidates if folder.exists()]

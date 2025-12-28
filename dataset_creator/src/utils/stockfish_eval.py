"""
Stockfish evaluation utilities for chess position analysis.

Provides centipawn evaluations for legal moves in a position, including optional
PV storage and fixed root-move probes for shallow/confirm passes (used to enrich
reasoning traces without requiring full MultiPV over every legal move).
"""

import chess
import chess.engine
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MoveEvaluation:
    """Evaluation of a single move."""
    uci: str
    san: str
    centipawn: int  # Centipawn score (positive = good for side to move)
    mate_in: Optional[int] = None  # Mate in N moves (positive = winning)
    pv_uci: List[str] = field(default_factory=list)  # Principal variation in UCI (truncated)
    
    def score_str(self) -> str:
        """Human-readable score string."""
        if self.mate_in is not None:
            return f"M{self.mate_in}" if self.mate_in > 0 else f"M{self.mate_in}"
        return f"{self.centipawn:+d}cp"
    
    def normalized_score(self, max_cp: int = 1000) -> float:
        """
        Normalize score to [-1, 1] range for reward shaping.
        Mate scores get ±1.0
        """
        if self.mate_in is not None:
            return 1.0 if self.mate_in > 0 else -1.0
        # Sigmoid-like clamping
        return max(-1.0, min(1.0, self.centipawn / max_cp))


@dataclass
class PositionAnalysis:
    """Complete analysis of a chess position."""
    fen: str
    move_evaluations: List[MoveEvaluation]
    best_move_uci: str
    best_move_san: str
    best_score_cp: int
    position_eval_cp: int = 0  # Overall position evaluation (optional)


class StockfishEvaluator:
    """
    Wrapper for Stockfish engine to evaluate positions and moves.
    
    Supports multi-threaded analysis for batch processing.
    """
    
    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 12,
        threads: int = 1,
        hash_mb: int = 128
    ):
        """
        Initialize Stockfish evaluator.
        
        Args:
            stockfish_path: Path to Stockfish binary. Auto-detects if None.
            depth: Search depth for analysis
            threads: Number of threads for Stockfish to use
            hash_mb: Hash table size in MB
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None
    
    @staticmethod
    def _find_stockfish() -> str:
        """Find Stockfish binary."""
        possible_paths = [
            '/usr/bin/stockfish',
            '/usr/games/stockfish',
            '/usr/local/bin/stockfish',
            '/opt/homebrew/bin/stockfish',
            'stockfish',  # In PATH
        ]
        
        import shutil
        sf_in_path = shutil.which('stockfish')
        if sf_in_path:
            return sf_in_path
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(
            "Stockfish not found. Install with: sudo apt install stockfish"
        )
    
    def _get_engine(self) -> chess.engine.SimpleEngine:
        """Get or create engine instance."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self._engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb,
            })
        return self._engine
    
    def close(self):
        """Close engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def analyze_position(
        self,
        board: chess.Board,
        multipv: Optional[int] = None,
        *,
        pv_max_len: int = 0,
        pv_store_top_k: int = 0,
        include_position_eval: bool = True,
    ) -> PositionAnalysis:
        """
        Analyze a position and return evaluations for all legal moves.
        
        Args:
            board: Chess position to analyze
            multipv: Number of principal variations (None = all legal moves)
            pv_max_len: If >0, store up to this many PV plies per move.
            pv_store_top_k: Store PVs only for the first K lines (best-first).
        
        Returns:
            PositionAnalysis with all move evaluations sorted best to worst
        """
        engine = self._get_engine()
        
        legal_moves = list(board.legal_moves)
        num_moves = len(legal_moves)
        
        if num_moves == 0:
            raise ValueError("No legal moves in position")
        
        # Use multipv to get all moves if not specified
        if multipv is None:
            multipv = num_moves
        else:
            multipv = min(multipv, num_moves)
        
        # Analyze with MultiPV
        result = engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=multipv
        )
        if isinstance(result, dict):
            result = [result]
        
        # Convert to MoveEvaluation objects
        move_evals = []
        
        pv_max_len = int(pv_max_len or 0)
        pv_store_top_k = int(pv_store_top_k or 0)

        for idx, info in enumerate(result):
            if 'pv' not in info or len(info['pv']) == 0:
                continue
            
            move = info['pv'][0]
            score = info.get('score')
            
            if score is None:
                continue
            
            # Get centipawn score from white's perspective, then adjust
            pov_score = score.pov(board.turn)
            
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                # Use large centipawn value for mate
                cp = 30000 if mate_in > 0 else -30000
            else:
                cp = pov_score.score()
                mate_in = None
            
            pv_uci: List[str] = []
            if pv_max_len > 0 and pv_store_top_k > 0 and idx < pv_store_top_k:
                try:
                    pv_uci = [m.uci() for m in (info.get("pv") or [])][:pv_max_len]
                except Exception:
                    pv_uci = []

            move_evals.append(
                MoveEvaluation(
                    uci=move.uci(),
                    san=board.san(move),
                    centipawn=cp,
                    mate_in=mate_in,
                    pv_uci=pv_uci,
                )
            )
        
        # Sort by centipawn (best first)
        move_evals.sort(key=lambda x: x.centipawn, reverse=True)
        
        pos_eval_cp = 0
        if include_position_eval:
            pos_info = engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth),
                multipv=1
            )
            if isinstance(pos_info, list):
                pos_info = pos_info[0] if pos_info else {}
            pos_score = (pos_info.get("score") or chess.engine.PovScore(chess.engine.Cp(0), board.turn)).pov(board.turn)
            pos_eval_cp = pos_score.score() if not pos_score.is_mate() else (30000 if pos_score.mate() > 0 else -30000)
        
        best = move_evals[0] if move_evals else None
        
        return PositionAnalysis(
            fen=board.fen(),
            move_evaluations=move_evals,
            best_move_uci=best.uci if best else "",
            best_move_san=best.san if best else "",
            best_score_cp=best.centipawn if best else 0,
            position_eval_cp=pos_eval_cp
        )

    def analyze_root_moves(
        self,
        board: chess.Board,
        root_moves: List[chess.Move],
        *,
        depth: int,
    ) -> Dict[str, int]:
        """
        Evaluate a fixed set of root moves at a given depth.

        This is used for lightweight shallow/confirm passes (e.g. trap detection)
        without requiring a full MultiPV over all legal moves.

        Returns:
            Mapping from UCI move -> centipawn evaluation (from side-to-move POV).
        """
        if not root_moves:
            return {}

        engine = self._get_engine()
        depth = int(depth)
        if depth <= 0:
            return {}

        multipv = min(len(root_moves), len(list(board.legal_moves)))
        if multipv <= 0:
            return {}

        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=multipv,
            root_moves=root_moves,
        )
        if isinstance(info, dict):
            infos = [info]
        else:
            infos = list(info or [])

        out: Dict[str, int] = {}
        for one in infos:
            pv = one.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            score = one.get("score")
            if score is None:
                continue
            pov_score = score.pov(board.turn)
            if pov_score.is_mate():
                mate_in = pov_score.mate() or 0
                cp = 30000 if mate_in > 0 else -30000
            else:
                cp = pov_score.score() or 0
            out[move.uci()] = int(cp)
        return out

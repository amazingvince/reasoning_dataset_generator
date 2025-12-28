"""
Chess LLM App - Modal deployment with vLLM and GPU Memory Snapshots
Serves amazingvince/chess_qwen3_4b_reasoning_v2 for chess games

Features:
- GPU memory snapshots for ~10x faster cold starts (45s -> 5s typical)
- Streaming LLM reasoning
- Play as White or Black
- Opening book support (12 openings)

Deploy with: modal deploy chess_app.py
Run locally: modal serve chess_app.py

GPU Snapshot Notes:
- First few cold starts will be slower as snapshots are created
- After snapshot creation, subsequent starts are dramatically faster
- Requires deployed app (snapshots don't work with `modal run`)
"""

import json
import os
import re
import socket
import subprocess
import modal

# --- Configuration ---
MODEL_NAME = os.environ.get("CHESS_MODEL_NAME", "amazingvince/chess_qwen3_4b_reasoning_v2")
GPU_TYPE = os.environ.get("CHESS_GPU", "T4")
N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000

# Snapshot version - change this to invalidate cached snapshots
SNAPSHOT_VERSION = os.environ.get("CHESS_SNAPSHOT_VERSION", "v1")

# --- Opening Book ---
OPENING_BOOK = {
    "starting": {
        "name": "Starting Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": [],
        "description": "Standard starting position"
    },
    "italian": {
        "name": "Italian Game",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
        "description": "1.e4 e5 2.Nf3 Nc6 3.Bc4 - Classical opening"
    },
    "sicilian": {
        "name": "Sicilian Defense",
        "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "moves": ["e2e4", "c7c5"],
        "description": "1.e4 c5 - Most popular response to e4"
    },
    "sicilian_najdorf": {
        "name": "Sicilian Najdorf",
        "fen": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
        "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
        "description": "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 - Bobby Fischer's favorite"
    },
    "queens_gambit": {
        "name": "Queen's Gambit",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
        "moves": ["d2d4", "d7d5", "c2c4"],
        "description": "1.d4 d5 2.c4 - Classic positional opening"
    },
    "kings_indian": {
        "name": "King's Indian Defense",
        "fen": "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5",
        "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
        "description": "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 - Dynamic counterattacking setup"
    },
    "french": {
        "name": "French Defense",
        "fen": "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
        "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"],
        "description": "1.e4 e6 2.d4 d5 3.e5 - Advance Variation"
    },
    "caro_kann": {
        "name": "Caro-Kann Defense",
        "fen": "rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
        "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"],
        "description": "1.e4 c6 2.d4 d5 3.e5 - Advance Variation"
    },
    "ruy_lopez": {
        "name": "Ruy Lopez",
        "fen": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
        "description": "1.e4 e5 2.Nf3 Nc6 3.Bb5 - The Spanish Game"
    },
    "london": {
        "name": "London System",
        "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
        "moves": ["d2d4", "d7d5", "c1f4", "g8f6", "g1f3"],
        "description": "1.d4 d5 2.Bf4 Nf6 3.Nf3 - Solid system for White"
    },
    "scandinavian": {
        "name": "Scandinavian Defense",
        "fen": "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        "moves": ["e2e4", "d7d5", "e4d5"],
        "description": "1.e4 d5 2.exd5 - Center Counter"
    },
    "english": {
        "name": "English Opening",
        "fen": "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
        "moves": ["c2c4"],
        "description": "1.c4 - Flexible flank opening"
    },
}

# --- Build the container image ---
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.8.5",
        "huggingface_hub[hf_transfer]",
        "flashinfer-python==0.2.2",
        "requests>=2.31.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # vLLM sleep mode for GPU snapshots
        "VLLM_SERVER_DEV_MODE": "1",
        # Improve torch compile compatibility with snapshots
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

# --- Create volumes for caching ---
hf_cache_vol = modal.Volume.from_name("chess-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("chess-vllm-cache", create_if_missing=True)

app = modal.App("chess-llm-game")


# --- Helper functions for vLLM server management ---
with vllm_image.imports():
    import requests as http_requests


def wait_for_server(proc: subprocess.Popen, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready by polling the health endpoint"""
    import time
    start = time.time()
    while time.time() - start < timeout:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            # Double-check with health endpoint
            try:
                resp = http_requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except:
                pass
        except OSError:
            pass
        
        # Check if process died
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM process exited with code {proc.returncode}")
        
        time.sleep(1)
    
    raise TimeoutError(f"vLLM server did not start within {timeout}s")


def sleep_vllm(level: int = 1):
    """Put vLLM server to sleep for snapshotting (offloads weights to CPU)"""
    http_requests.post(
        f"http://localhost:{VLLM_PORT}/sleep?level={level}",
        timeout=60
    ).raise_for_status()


def wake_vllm():
    """Wake up vLLM server after snapshot restore"""
    http_requests.post(
        f"http://localhost:{VLLM_PORT}/wake_up",
        timeout=60
    ).raise_for_status()


def warmup_vllm():
    """Warmup the server to capture JIT compilation artifacts in snapshot"""
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "e4"}],
        "max_tokens": 32,
        "temperature": 0.7,
    }
    # Run a few warmup requests
    for _ in range(3):
        http_requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=120,
        ).raise_for_status()


# --- vLLM Server with GPU Memory Snapshots ---
@app.cls(
    image=vllm_image,
    gpu=GPU_TYPE,  # T4 is sufficient for 4B model, cost-effective
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Enable GPU memory snapshots for faster cold starts
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    allow_concurrent_inputs=10,
)
class ChessLLMServer:
    """
    vLLM server with GPU memory snapshot support.
    
    The snapshot workflow:
    1. start_and_snapshot (snap=True): Start vLLM, warmup, then sleep
    2. Modal creates a GPU memory snapshot
    3. On restore, wake_after_restore (snap=False) wakes the server
    4. Subsequent cold starts skip model loading entirely (~10x faster)
    """
    
    @modal.enter(snap=True)
    def start_and_snapshot(self):
        """Start vLLM server, warmup for JIT compilation, then sleep for snapshot"""
        print(f"[{SNAPSHOT_VERSION}] Starting vLLM server for {MODEL_NAME}...")
        
        cmd = [
            "vllm", "serve",
            MODEL_NAME,
            "--served-model-name", "llm",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "4096",
            "--trust-remote-code",
            # Snapshot-friendly settings
            "--enable-sleep-mode",  # Required for sleep/wake
            "--max-num-seqs", "4",  # Smaller KV cache for faster snapshot
            "--max-num-batched-tokens", "4096",
        ]
        
        print(f"Command: {' '.join(cmd)}")
        self.vllm_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        print("Waiting for vLLM server to be ready...")
        wait_for_server(self.vllm_proc, timeout=300)
        print("✓ Server is ready!")
        
        print("Warming up model for JIT compilation...")
        warmup_vllm()
        print("✓ Warmup complete!")
        
        print("Putting server to sleep for snapshot...")
        sleep_vllm()
        print("✓ Server is asleep, ready for GPU memory snapshot!")
    
    @modal.enter(snap=False)
    def wake_after_restore(self):
        """Wake server after snapshot restore - this runs on every restored container"""
        print("Waking up vLLM server after snapshot restore...")
        wake_vllm()
        # Quick health check
        wait_for_server(self.vllm_proc, timeout=30)
        print("✓ Server is awake and ready!")
    
    @modal.method()
    def generate(self, prompt: str) -> str:
        """Generate completion from the LLM (non-streaming)"""
        full_prompt = prompt.rstrip() + "\n<think>"
        
        payload = {
            "model": "llm",
            "prompt": full_prompt,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "stop": ["</uci_move>"],
        }
        
        response = http_requests.post(
            f"http://localhost:{VLLM_PORT}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        
        result = response.json()
        text = result["choices"][0]["text"]
        
        # Add back closing tag if truncated
        if "<uci_move>" in text and "</uci_move>" not in text:
            text += "</uci_move>"
        
        return "<think>" + text
    
    @modal.method()
    def generate_streaming(self, prompt: str):
        """Generate with streaming for real-time UI updates"""
        full_prompt = prompt.rstrip() + "\n<think>"
        
        payload = {
            "model": "llm",
            "prompt": full_prompt,
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "stream": True,
            "stop": ["</uci_move>"],
        }
        
        response = http_requests.post(
            f"http://localhost:{VLLM_PORT}/v1/completions",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        
        full_text = "<think>"
        yield full_text
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        text = chunk["choices"][0].get("text", "")
                        if text:
                            full_text += text
                            yield text
                    except json.JSONDecodeError:
                        continue
        
        # Add closing tag if needed
        if "<uci_move>" in full_text and "</uci_move>" not in full_text:
            yield "</uci_move>"
    
    @modal.exit()
    def shutdown(self):
        """Clean shutdown of vLLM server"""
        if hasattr(self, 'vllm_proc') and self.vllm_proc:
            print("Shutting down vLLM server...")
            self.vllm_proc.terminate()
            try:
                self.vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_proc.kill()


# --- FastAPI Web Server ---
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "python-chess",
        "uvicorn>=0.34.0",
        "pydantic>=2.0.0",
        "sse-starlette>=2.0.0",
    )
)


@app.function(image=web_image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from sse_starlette.sse import EventSourceResponse
    from pydantic import BaseModel
    import chess
    import asyncio
    import uuid
    
    web_app = FastAPI(title="Chess LLM Game")
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Game state storage (in-memory)
    games: dict = {}
    
    class GameConfig(BaseModel):
        player_color: str = "white"
        opening: str = "starting"
    
    class MoveRequest(BaseModel):
        game_id: str
        move: str
    
    class GameState:
        def __init__(self, player_color: str = "white", opening: str = "starting"):
            self.board = chess.Board()
            self.player_color = player_color
            self.opening = opening
            self.move_history = []
            self.reasoning_history = []
            
            # Apply opening moves
            if opening in OPENING_BOOK:
                for uci_move in OPENING_BOOK[opening]["moves"]:
                    move = chess.Move.from_uci(uci_move)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.move_history.append(uci_move)
        
        def get_fen(self) -> str:
            return self.board.fen()
        
        def get_legal_moves_uci(self) -> list[str]:
            return [move.uci() for move in self.board.legal_moves]
        
        def make_move(self, uci_move: str) -> bool:
            try:
                move = chess.Move.from_uci(uci_move)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.move_history.append(uci_move)
                    return True
                return False
            except:
                return False
        
        def is_game_over(self) -> dict:
            return {
                "is_over": self.board.is_game_over(),
                "is_checkmate": self.board.is_checkmate(),
                "is_stalemate": self.board.is_stalemate(),
                "is_draw": self.board.is_insufficient_material() or 
                          self.board.can_claim_fifty_moves() or 
                          self.board.can_claim_threefold_repetition(),
                "winner": "white" if self.board.is_checkmate() and not self.board.turn else 
                         "black" if self.board.is_checkmate() else None
            }
        
        def is_player_turn(self) -> bool:
            if self.player_color == "white":
                return self.board.turn  # True = white's turn
            else:
                return not self.board.turn
        
        def build_llm_prompt(self) -> str:
            fen = self.get_fen()
            legal_moves = " ".join(sorted(self.get_legal_moves_uci()))
            
            return f"""You are an expert chess player. Choose the best move.
FEN: {fen}
Legal moves (UCI): {legal_moves}

Rules:
- Put all reasoning inside <think>...</think>.
- Output exactly one <uci_move>...</uci_move> tag with a single move copied from the legal moves list (no spaces).
- Do not output anything after the closing </uci_move>.
- Do not output "resign".

Output format:
<think>...</think>
<uci_move>...</uci_move>
"""
    
    def parse_llm_response(response: str) -> tuple:
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None
        
        move_match = re.search(r'<uci_move>(\S+)</uci_move>', response)
        move = move_match.group(1).strip() if move_match else None
        
        return reasoning, move
    
    @web_app.get("/")
    async def serve_ui():
        return HTMLResponse(content=get_chess_html())
    
    @web_app.get("/api/openings")
    async def get_openings():
        return {key: {"name": val["name"], "description": val["description"]} 
                for key, val in OPENING_BOOK.items()}
    
    @web_app.post("/api/new-game")
    async def new_game(config: GameConfig = GameConfig()):
        game_id = str(uuid.uuid4())[:8]
        games[game_id] = GameState(config.player_color, config.opening)
        
        opening_name = OPENING_BOOK.get(config.opening, OPENING_BOOK["starting"])["name"]
        
        return {
            "game_id": game_id,
            "fen": games[game_id].get_fen(),
            "legal_moves": games[game_id].get_legal_moves_uci(),
            "player_color": config.player_color,
            "opening_name": opening_name,
            "move_history": games[game_id].move_history,
            "is_player_turn": games[game_id].is_player_turn()
        }
    
    @web_app.get("/api/game/{game_id}")
    async def get_game_state(game_id: str):
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = games[game_id]
        return {
            "fen": game.get_fen(),
            "legal_moves": game.get_legal_moves_uci(),
            "game_over": game.is_game_over(),
            "move_history": game.move_history,
            "is_player_turn": game.is_player_turn(),
            "player_color": game.player_color
        }
    
    @web_app.get("/api/llm-move/{game_id}")
    async def get_llm_move_stream(game_id: str):
        """Stream the LLM's move with reasoning"""
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = games[game_id]
        
        if game.is_game_over()["is_over"]:
            raise HTTPException(status_code=400, detail="Game is over")
        
        async def generate():
            try:
                llm = ChessLLMServer()
                prompt = game.build_llm_prompt()
                
                full_response = ""
                
                # Stream the response
                for chunk in llm.generate_streaming.remote_gen(prompt):
                    full_response += chunk
                    yield {
                        "event": "chunk",
                        "data": json.dumps({"text": chunk, "done": False})
                    }
                    await asyncio.sleep(0.02)
                
                # Parse the complete response
                reasoning, llm_move = parse_llm_response(full_response)
                
                # Make the move
                move_success = False
                final_move = None
                
                if llm_move and game.make_move(llm_move):
                    move_success = True
                    final_move = llm_move
                    game.reasoning_history.append(reasoning)
                else:
                    # Fallback to random legal move
                    import random
                    legal = game.get_legal_moves_uci()
                    if legal:
                        fallback = random.choice(legal)
                        game.make_move(fallback)
                        final_move = fallback
                        move_success = True
                
                # Send final state
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "done": True,
                        "success": move_success,
                        "move": final_move,
                        "reasoning": reasoning,
                        "fen": game.get_fen(),
                        "legal_moves": game.get_legal_moves_uci(),
                        "game_over": game.is_game_over(),
                        "is_player_turn": game.is_player_turn()
                    })
                }
                
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
        
        return EventSourceResponse(generate())
    
    @web_app.post("/api/move")
    async def make_player_move(request: MoveRequest):
        if request.game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = games[request.game_id]
        
        if not game.is_player_turn():
            return {"success": False, "error": "Not your turn"}
        
        if not game.make_move(request.move):
            return {
                "success": False,
                "fen": game.get_fen(),
                "legal_moves": game.get_legal_moves_uci(),
                "game_over": game.is_game_over(),
                "error": "Invalid move"
            }
        
        return {
            "success": True,
            "fen": game.get_fen(),
            "legal_moves": game.get_legal_moves_uci(),
            "game_over": game.is_game_over(),
            "is_player_turn": game.is_player_turn()
        }
    
    @web_app.post("/api/reset/{game_id}")
    async def reset_game(game_id: str):
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = games[game_id]
        player_color = game.player_color
        opening = game.opening
        games[game_id] = GameState(player_color, opening)
        
        return {
            "success": True, 
            "fen": games[game_id].get_fen(),
            "move_history": games[game_id].move_history,
            "is_player_turn": games[game_id].is_player_turn()
        }
    
    return web_app


def get_chess_html() -> str:
    """Return the complete chess UI HTML"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess vs LLM</title>
    
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.13.4/chess.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --success: #22c55e;
            --warning: #f59e0b;
            --text: #e2e8f0;
            --text-muted: #64748b;
            --border: rgba(255, 255, 255, 0.08);
        }
        
        body {
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(34, 197, 94, 0.05) 0%, transparent 50%);
            min-height: 100vh;
            color: var(--text);
        }
        
        .app { max-width: 1600px; margin: 0 auto; padding: 1.5rem; }
        
        header { text-align: center; padding: 1rem 0 2rem; }
        
        h1 {
            font-family: 'Instrument Serif', serif;
            font-size: 3rem;
            font-weight: 400;
            font-style: italic;
            background: linear-gradient(135deg, #fff 0%, #6366f1 50%, #22c55e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }
        
        .subtitle { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 3px; }
        
        .badge {
            display: inline-block;
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.65rem;
            margin-left: 0.5rem;
        }
        
        .game-layout {
            display: grid;
            grid-template-columns: 280px 1fr 380px;
            gap: 1.5rem;
            align-items: start;
        }
        
        @media (max-width: 1200px) {
            .game-layout { grid-template-columns: 1fr; max-width: 500px; margin: 0 auto; }
        }
        
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
        }
        
        .panel-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        
        .panel-header .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 8px var(--accent-glow);
        }
        
        .panel-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: var(--text-muted); }
        
        #board {
            width: 100% !important;
            max-width: 500px;
            margin: 0 auto;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 60px rgba(99, 102, 241, 0.15);
        }
        
        .status-bar {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.875rem 1.25rem;
            text-align: center;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .status-bar.thinking { border-color: var(--warning); background: rgba(245, 158, 11, 0.1); }
        .status-bar.your-turn { border-color: var(--success); background: rgba(34, 197, 94, 0.1); }
        
        .spinner {
            width: 14px; height: 14px;
            border: 2px solid var(--warning);
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .controls { display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; }
        
        .btn {
            font-family: inherit;
            font-size: 0.75rem;
            padding: 0.625rem 1.25rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: transparent;
            color: var(--text);
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn:hover:not(:disabled) { border-color: var(--accent); background: rgba(99, 102, 241, 0.1); }
        .btn-primary { background: var(--accent); border-color: var(--accent); }
        .btn-primary:hover:not(:disabled) { background: #5558e3; transform: translateY(-1px); box-shadow: 0 4px 12px var(--accent-glow); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .form-group label {
            display: block;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        
        select {
            width: 100%;
            padding: 0.625rem;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-family: inherit;
            font-size: 0.8rem;
            cursor: pointer;
        }
        
        select:focus { outline: none; border-color: var(--accent); }
        
        .color-toggle { display: flex; gap: 0.5rem; }
        
        .color-btn {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
            font-size: 0.75rem;
        }
        
        .color-btn.active { border-color: var(--accent); background: rgba(99, 102, 241, 0.15); color: var(--text); }
        .color-btn:hover:not(.active) { border-color: var(--text-muted); }
        
        .move-history { max-height: 300px; overflow-y: auto; font-size: 0.8rem; }
        .move-history::-webkit-scrollbar { width: 4px; }
        .move-history::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
        
        .move-pair {
            display: grid;
            grid-template-columns: 2rem 1fr 1fr;
            gap: 0.5rem;
            padding: 0.375rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        .move-num { color: var(--text-muted); }
        .white-move { color: #fbbf24; }
        .black-move { color: #a78bfa; }
        
        .reasoning-box {
            background: var(--bg-dark);
            border-radius: 8px;
            padding: 1rem;
            max-height: 450px;
            overflow-y: auto;
            font-size: 0.8rem;
            line-height: 1.7;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .reasoning-box::-webkit-scrollbar { width: 4px; }
        .reasoning-box::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }
        
        .streaming-cursor {
            display: inline-block;
            width: 8px; height: 14px;
            background: var(--accent);
            margin-left: 2px;
            animation: blink 0.8s infinite;
        }
        
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        
        .empty-state { color: var(--text-muted); font-style: italic; text-align: center; padding: 2rem 1rem; }
        
        .game-over-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            backdrop-filter: blur(8px);
        }
        
        .game-over-modal {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 2.5rem 3rem;
            border-radius: 16px;
            text-align: center;
        }
        
        .game-over-title { font-family: 'Instrument Serif', serif; font-size: 2rem; font-style: italic; margin-bottom: 0.5rem; }
        .game-over-sub { color: var(--text-muted); margin-bottom: 1.5rem; }
        .opening-desc { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-style: italic; }
        .config-section { display: flex; flex-direction: column; gap: 1rem; }
        .board-section { display: flex; flex-direction: column; gap: 1rem; }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState, useEffect, useRef } = React;
        
        const OPENINGS = {
            starting: { name: "Starting Position", description: "Standard starting position" },
            italian: { name: "Italian Game", description: "1.e4 e5 2.Nf3 Nc6 3.Bc4" },
            sicilian: { name: "Sicilian Defense", description: "1.e4 c5" },
            sicilian_najdorf: { name: "Sicilian Najdorf", description: "Bobby Fischer's favorite" },
            queens_gambit: { name: "Queen's Gambit", description: "1.d4 d5 2.c4" },
            kings_indian: { name: "King's Indian", description: "Dynamic counterattack" },
            french: { name: "French Defense", description: "1.e4 e6 2.d4 d5 3.e5" },
            caro_kann: { name: "Caro-Kann", description: "1.e4 c6 2.d4 d5 3.e5" },
            ruy_lopez: { name: "Ruy Lopez", description: "The Spanish Game" },
            london: { name: "London System", description: "Solid for White" },
            scandinavian: { name: "Scandinavian", description: "1.e4 d5 2.exd5" },
            english: { name: "English Opening", description: "1.c4" },
        };
        
        function ChessGame() {
            const [gameId, setGameId] = useState(null);
            const [status, setStatus] = useState('Configure your game and click "New Game"');
            const [isThinking, setIsThinking] = useState(false);
            const [isPlayerTurn, setIsPlayerTurn] = useState(true);
            const [reasoning, setReasoning] = useState('');
            const [isStreaming, setIsStreaming] = useState(false);
            const [moveHistory, setMoveHistory] = useState([]);
            const [gameOver, setGameOver] = useState(null);
            const [playerColor, setPlayerColor] = useState('white');
            const [selectedOpening, setSelectedOpening] = useState('starting');
            
            const boardRef = useRef(null);
            const chessRef = useRef(null);
            
            useEffect(() => {
                chessRef.current = new Chess();
                boardRef.current = Chessboard('board', {
                    draggable: true,
                    position: 'start',
                    orientation: 'white',
                    onDragStart, onDrop, onSnapEnd,
                    pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
                });
                $(window).resize(() => boardRef.current?.resize());
                return () => boardRef.current?.destroy();
            }, []);
            
            useEffect(() => { boardRef.current?.orientation(playerColor); }, [playerColor]);
            
            const onDragStart = (source, piece) => {
                if (!gameId || isThinking || !isPlayerTurn || chessRef.current.game_over()) return false;
                const isWhitePiece = piece.search(/^w/) !== -1;
                if (playerColor === 'white' && !isWhitePiece) return false;
                if (playerColor === 'black' && isWhitePiece) return false;
                return true;
            };
            
            const onDrop = async (source, target) => {
                const move = chessRef.current.move({ from: source, to: target, promotion: 'q' });
                if (!move) return 'snapback';
                
                const uciMove = source + target + (move.promotion || '');
                try {
                    const response = await fetch('/api/move', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ game_id: gameId, move: uciMove })
                    });
                    const data = await response.json();
                    if (data.success) {
                        updateMoveHistory(uciMove, null);
                        if (data.game_over.is_over) {
                            setGameOver(data.game_over);
                            setStatus('Game Over!');
                        } else {
                            setIsPlayerTurn(false);
                            triggerLLMMove();
                        }
                    } else {
                        chessRef.current.undo();
                        boardRef.current.position(chessRef.current.fen());
                    }
                } catch {
                    chessRef.current.undo();
                    boardRef.current.position(chessRef.current.fen());
                }
            };
            
            const onSnapEnd = () => boardRef.current.position(chessRef.current.fen());
            
            const updateMoveHistory = (whiteMove, blackMove) => {
                setMoveHistory(prev => {
                    const h = [...prev];
                    if (playerColor === 'white') {
                        if (whiteMove && !blackMove) h.push({ white: whiteMove, black: null });
                        else if (blackMove && h.length && h[h.length-1].black === null) h[h.length-1].black = blackMove;
                    } else {
                        if (blackMove && !whiteMove) {
                            if (h.length && h[h.length-1].black === null) h[h.length-1].black = blackMove;
                            else h.push({ white: null, black: blackMove });
                        } else if (whiteMove) h.push({ white: whiteMove, black: null });
                    }
                    return h;
                });
            };
            
            const triggerLLMMove = async () => {
                setIsThinking(true);
                setIsStreaming(true);
                setStatus('LLM is thinking...');
                setReasoning('');
                
                try {
                    const es = new EventSource(`/api/llm-move/${gameId}`);
                    let fullText = '';
                    
                    es.addEventListener('chunk', e => {
                        const d = JSON.parse(e.data);
                        fullText += d.text;
                        const m = fullText.match(/<think>([\s\S]*?)(<\/think>|$)/);
                        if (m) setReasoning(m[1]);
                    });
                    
                    es.addEventListener('complete', e => {
                        const d = JSON.parse(e.data);
                        es.close();
                        setIsStreaming(false);
                        setIsThinking(false);
                        if (d.success && d.move) {
                            chessRef.current.load(d.fen);
                            boardRef.current.position(d.fen);
                            playerColor === 'white' ? updateMoveHistory(null, d.move) : updateMoveHistory(d.move, null);
                            if (d.reasoning) setReasoning(d.reasoning);
                            if (d.game_over.is_over) { setGameOver(d.game_over); setStatus('Game Over!'); }
                            else { setIsPlayerTurn(true); setStatus(`Your turn (${playerColor})`); }
                        }
                    });
                    
                    es.addEventListener('error', () => { es.close(); setIsStreaming(false); setIsThinking(false); setStatus('Error - try again'); });
                } catch { setIsStreaming(false); setIsThinking(false); setStatus('Error'); }
            };
            
            const startNewGame = async () => {
                setIsThinking(true);
                setStatus('Starting new game...');
                try {
                    const r = await fetch('/api/new-game', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ player_color: playerColor, opening: selectedOpening })
                    });
                    const d = await r.json();
                    setGameId(d.game_id);
                    setMoveHistory([]);
                    setReasoning('');
                    setGameOver(null);
                    chessRef.current.load(d.fen);
                    boardRef.current.position(d.fen);
                    boardRef.current.orientation(playerColor);
                    if (d.move_history?.length) {
                        const pairs = [];
                        for (let i = 0; i < d.move_history.length; i += 2)
                            pairs.push({ white: d.move_history[i] || null, black: d.move_history[i+1] || null });
                        setMoveHistory(pairs);
                    }
                    setIsPlayerTurn(d.is_player_turn);
                    setIsThinking(false);
                    d.is_player_turn ? setStatus(`Your turn (${playerColor})`) : triggerLLMMove();
                } catch { setIsThinking(false); setStatus('Error starting game'); }
            };
            
            const getGameOverMessage = () => {
                if (!gameOver) return '';
                if (gameOver.is_checkmate) return gameOver.winner === playerColor ? '🎉 Victory!' : '😔 Defeat';
                if (gameOver.is_stalemate) return '🤝 Stalemate';
                return '🤝 Draw';
            };
            
            return (
                <div className="app">
                    <header>
                        <h1>Chess vs LLM</h1>
                        <p className="subtitle">amazingvince/chess_qwen3_4b_reasoning_v2<span className="badge">⚡ GPU Snapshots</span></p>
                    </header>
                    
                    <div className="game-layout">
                        <div>
                            <div className="panel config-section">
                                <div className="panel-header"><div className="dot"></div><span className="panel-title">Game Setup</span></div>
                                <div className="form-group">
                                    <label>Play as</label>
                                    <div className="color-toggle">
                                        <button className={`color-btn ${playerColor==='white'?'active':''}`} onClick={()=>setPlayerColor('white')} disabled={!!gameId}>♔ White</button>
                                        <button className={`color-btn ${playerColor==='black'?'active':''}`} onClick={()=>setPlayerColor('black')} disabled={!!gameId}>♚ Black</button>
                                    </div>
                                </div>
                                <div className="form-group">
                                    <label>Opening</label>
                                    <select value={selectedOpening} onChange={e=>setSelectedOpening(e.target.value)} disabled={!!gameId}>
                                        {Object.entries(OPENINGS).map(([k,v])=><option key={k} value={k}>{v.name}</option>)}
                                    </select>
                                    <div className="opening-desc">{OPENINGS[selectedOpening]?.description}</div>
                                </div>
                            </div>
                            <div className="panel" style={{marginTop:'1rem'}}>
                                <div className="panel-header"><div className="dot" style={{background:'#fbbf24'}}></div><span className="panel-title">Moves</span></div>
                                {moveHistory.length===0 ? <div className="empty-state">No moves yet</div> : (
                                    <div className="move-history">
                                        {moveHistory.map((p,i)=><div key={i} className="move-pair"><span className="move-num">{i+1}.</span><span className="white-move">{p.white||'...'}</span><span className="black-move">{p.black||''}</span></div>)}
                                    </div>
                                )}
                            </div>
                        </div>
                        
                        <div className="board-section">
                            <div className={`status-bar ${isThinking?'thinking':isPlayerTurn?'your-turn':''}`}>
                                {isThinking && <div className="spinner"></div>}{status}
                            </div>
                            <div id="board"></div>
                            <div className="controls">
                                <button className="btn btn-primary" onClick={startNewGame} disabled={isThinking}>New Game</button>
                                <button className="btn" onClick={()=>{setGameId(null);setMoveHistory([]);setReasoning('');setGameOver(null);chessRef.current.reset();boardRef.current.position('start');setStatus('Configure and click "New Game"');}} disabled={isThinking}>Reset</button>
                            </div>
                        </div>
                        
                        <div className="panel">
                            <div className="panel-header"><div className="dot" style={{background:'#22c55e'}}></div><span className="panel-title">LLM Reasoning</span></div>
                            <div className="reasoning-box">
                                {reasoning ? <>{reasoning}{isStreaming && <span className="streaming-cursor"></span>}</> : <span className="empty-state">LLM thinking will stream here...</span>}
                            </div>
                        </div>
                    </div>
                    
                    {gameOver?.is_over && (
                        <div className="game-over-overlay" onClick={startNewGame}>
                            <div className="game-over-modal" onClick={e=>e.stopPropagation()}>
                                <div className="game-over-title">{getGameOverMessage()}</div>
                                <div className="game-over-sub">Click to play again</div>
                                <button className="btn btn-primary" onClick={startNewGame}>New Game</button>
                            </div>
                        </div>
                    )}
                </div>
            );
        }
        
        ReactDOM.render(<ChessGame />, document.getElementById('root'));
    </script>
</body>
</html>'''


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  Chess LLM Game with GPU Memory Snapshots")
    print("=" * 60)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  Snapshot Version: {SNAPSHOT_VERSION}")
    print("\n📦 DEPLOY:")
    print("    modal deploy chess_app.py")
    print("\n🔧 DEV MODE:")
    print("    modal serve chess_app.py")
    print("\n✨ FEATURES:")
    print("  • ⚡ GPU snapshots for ~10x faster cold starts")
    print("  • 🔄 Streaming LLM reasoning")
    print("  • ⚪⚫ Play as White or Black")
    print("  • 📖 12 opening positions")
    print("\n💡 NOTE: First few requests create snapshots (slower).")
    print("   Subsequent cold starts will be dramatically faster!")
    print("\n" + "=" * 60)

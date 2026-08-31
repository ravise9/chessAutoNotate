from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chess_ai import identify_move

app = FastAPI(title="Chess Move Notation")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@dataclass
class GameState:
    board: chess.Board = field(default_factory=chess.Board)
    game: chess.pgn.Game = field(default_factory=chess.pgn.Game)
    node: chess.pgn.GameNode = field(default=None)
    moves_san: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.node is None:
            self.node = self.game

    def reset(self):
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.node = self.game
        self.moves_san = []


state = GameState()


def _state_response() -> dict:
    return {
        "fen": state.board.fen(),
        "turn": "white" if state.board.turn == chess.WHITE else "black",
        "move_number": state.board.fullmove_number,
        "half_moves": len(state.moves_san),
        "moves": state.moves_san,
        "game_over": state.board.is_game_over(),
        "outcome": _outcome_text(),
    }


def _outcome_text() -> Optional[str]:
    outcome = state.board.outcome()
    if outcome is None:
        return None
    if outcome.winner == chess.WHITE:
        return "White wins"
    if outcome.winner == chess.BLACK:
        return "Black wins"
    return "Draw"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def get_state():
    return _state_response()


@app.post("/api/new-game")
async def new_game():
    state.reset()
    return {"ok": True, **_state_response()}


@app.post("/api/move")
async def add_move(image: UploadFile = File(...)):
    if state.board.is_game_over():
        raise HTTPException(400, "Game is already over. Start a new game.")

    content_type = image.content_type or "image/jpeg"
    if content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(400, f"Unsupported image type: {content_type}. Use JPEG or PNG.")

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 20 MB).")

    try:
        move = identify_move(state.board, image_bytes, content_type)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    san = state.board.san(move)
    state.node = state.node.add_variation(move)
    state.board.push(move)
    state.moves_san.append(san)

    return {
        "move": san,
        "uci": move.uci(),
        **_state_response(),
    }


@app.post("/api/end-game")
async def end_game():
    if not state.moves_san:
        raise HTTPException(400, "No moves have been played yet.")

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn = state.game.accept(exporter)
    return {"pgn": pgn, **_state_response()}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional

import anthropic
import chess

client = anthropic.Anthropic()

SYSTEM = (
    "You are a chess expert analyzing board photographs. "
    "You always respond with valid JSON only — no prose, no markdown fences."
)

PROMPT_TEMPLATE = """The chess board was in this FEN position before the move:
{fen}

The image shows the board AFTER a move was just played. Examine the image carefully:
- Compare piece positions to the given FEN
- Identify which piece moved and to where
- Account for captures, castling, and en passant

Return ONLY this JSON (no explanation, no code fences):
{{"from": "e2", "to": "e4", "promotion": null}}

"promotion" is null unless a pawn reached the back rank — then give the piece letter (q, r, b, or n).
"""

RETRY_PROMPT_TEMPLATE = """Look again at this chess board image.

The previous FEN (before the move) was:
{fen}

Legal moves from this position include (UCI format):
{legal_moves}

Which of these moves was just played? Return ONLY JSON:
{{"from": "e2", "to": "e4", "promotion": null}}
"""


def _encode_image(image_bytes: bytes, media_type: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
        },
    }


def _parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _build_messages(fen: str, image_bytes: bytes, media_type: str, retry: bool, board: chess.Board) -> list:
    if retry:
        legal_uci = [m.uci() for m in board.legal_moves][:30]
        prompt = RETRY_PROMPT_TEMPLATE.format(fen=fen, legal_moves=", ".join(legal_uci))
    else:
        prompt = PROMPT_TEMPLATE.format(fen=fen)

    return [{
        "role": "user",
        "content": [
            _encode_image(image_bytes, media_type),
            {"type": "text", "text": prompt},
        ],
    }]


def identify_move(board: chess.Board, image_bytes: bytes, media_type: str = "image/jpeg") -> chess.Move:
    """
    Send the board's current FEN + an image to Claude and return the identified chess.Move.
    Raises ValueError if the move cannot be identified or is illegal.
    """
    fen = board.fen()

    for attempt in range(2):
        messages = _build_messages(fen, image_bytes, media_type, retry=(attempt == 1), board=board)
        with client.messages.stream(
            model="claude-opus-5",
            max_tokens=256,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        parsed = _parse_json_response(text)
        if not parsed:
            continue

        from_sq_str = parsed.get("from", "").lower().strip()
        to_sq_str = parsed.get("to", "").lower().strip()
        promotion_str = parsed.get("promotion")

        try:
            from_sq = chess.parse_square(from_sq_str)
            to_sq = chess.parse_square(to_sq_str)
        except (ValueError, KeyError):
            continue

        promotion = None
        if promotion_str:
            promotion = chess.Piece.from_symbol(promotion_str.upper()).piece_type

        try:
            move = board.find_move(from_sq, to_sq, promotion=promotion)
            return move
        except chess.IllegalMoveError:
            continue

    raise ValueError(
        f"Could not identify a legal move from the image. "
        f"Claude returned: {text!r}. Please re-upload a clearer photo."
    )

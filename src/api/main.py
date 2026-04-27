from __future__ import annotations

import asyncio
from pathlib import Path
from queue import Empty, Queue
import random
import threading
from datetime import datetime
from typing import List

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.model.dqn import DQNNet
from src.model.dqn import legal_mask_from_state, masked_argmax, masked_softmax
from src.rl import DQNTrainer, TrainerConfig


app = FastAPI(title="Tic-Tac-Toe DQN API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
trainer = DQNTrainer(TrainerConfig())
stop_event = threading.Event()
training_status = {"running": False, "episodes_done": 0, "episodes_total": 0}
metrics_queue: Queue[dict] = Queue()
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
app.state.train_thread = None
app.state.stop_behavior = "save"
active_model_id: str | None = None
model_cache: dict[str, DQNNet] = {}


class TrainRequest(BaseModel):
    episodes: int = Field(default=100, ge=1, le=100_000)
    lr: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=64, ge=8, le=2048)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay_steps: int = Field(default=5_000, ge=1)
    target_sync_steps: int = Field(default=100, ge=1)


class BoardRequest(BaseModel):
    state: List[float] = Field(min_length=9, max_length=9)
    temperature: float = Field(default=1.0, ge=0.0)
    tau: float | None = Field(default=None, ge=0.0)
    policy: str = Field(default="model")
    model_id: str | None = None
    search_depth: int = Field(default=1, ge=1, le=6)


class TournamentRequest(BaseModel):
    games: int = Field(default=50, ge=1, le=10_000)
    x_policy: str = Field(default="model")
    o_policy: str = Field(default="random")
    x_model_id: str | None = None
    o_model_id: str | None = None
    temperature: float = Field(default=1.0, ge=0.0)
    search_depth: int = Field(default=1, ge=1, le=6)


class LeaderboardRequest(BaseModel):
    games_per_side: int = Field(default=50, ge=1, le=1000)


class MatrixRequest(BaseModel):
    games: int = Field(default=20, ge=1, le=500)


class ModelLoadRequest(BaseModel):
    model_id: str


class ModelDeleteRequest(BaseModel):
    model_id: str


WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _winner(board: List[float]) -> int:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def _draw(board: List[float]) -> bool:
    return all(v != 0.0 for v in board) and _winner(board) == 0


def _legal_actions(state: List[float]) -> List[int]:
    return [i for i, v in enumerate(state) if v == 0.0]


def _model_path(model_id: str) -> Path:
    return MODEL_DIR / f"{model_id}.pt"


def _list_model_entries() -> list[dict]:
    entries = []
    for p in sorted(MODEL_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        last_loss = None
        last_val_loss = None
        global_step = None
        try:
            ckpt = torch.load(p, map_location="cpu")
            if isinstance(ckpt, dict):
                if "last_loss" in ckpt:
                    last_loss = float(ckpt.get("last_loss", 0.0))
                if "last_val_loss" in ckpt:
                    last_val_loss = float(ckpt.get("last_val_loss", 0.0))
                if "global_step" in ckpt:
                    global_step = int(ckpt.get("global_step", 0))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            # Keep listing robust even if one checkpoint is corrupted/old format.
            pass
        entries.append(
            {
                "id": p.stem,
                "filename": p.name,
                "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "last_loss": last_loss,
                "last_val_loss": last_val_loss,
                "global_step": global_step,
            }
        )
    return entries


def _get_model_net(model_id: str | None) -> DQNNet:
    if not model_id:
        return trainer.online
    if model_id in model_cache:
        return model_cache[model_id]
    p = _model_path(model_id)
    if not p.exists():
        return trainer.online
    ckpt = torch.load(p, map_location=trainer.device)
    net = DQNNet().to(trainer.device)
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()
    model_cache[model_id] = net
    return net


def _model_action_with_probs(
    state: List[float],
    temperature: float,
    model_id: str | None = None,
    player: int = 1,
) -> tuple[int, list[float], list[float]]:
    # Network is trained from +1 perspective, so normalize board for current player.
    normalized_state = [v * player for v in state]
    tensor_state = torch.tensor([normalized_state], dtype=torch.float32, device=trainer.device)
    net = _get_model_net(model_id)
    with torch.no_grad():
        q = net(tensor_state)
        mask = legal_mask_from_state(tensor_state)
        # tau == 0 means pure greedy argmax over legal moves.
        if temperature <= 0.0:
            best_idx = int(masked_argmax(q, mask)[0].item())
            probs = torch.zeros_like(q)
            probs[0, best_idx] = 1.0
        else:
            probs = masked_softmax(q, mask, tau=temperature)
        # Convert q-values back to global X-perspective for consistent display.
        q_cpu = [float(v) * player for v in q[0].cpu().tolist()]
        p = probs[0].cpu().tolist()
    legal = _legal_actions(state)
    ranked = sorted(legal, key=lambda idx: p[idx], reverse=True)
    return ranked[0], q_cpu, p


def _state_value_for_player(state: List[float], player: int, model_id: str | None) -> float:
    # Network is trained from +1 perspective, so normalize when player is -1.
    normalized = [v * player for v in state]
    action, q_cpu, _ = _model_action_with_probs(normalized, temperature=1.0, model_id=model_id, player=1)
    value = q_cpu[action]
    return value * player


def _minimax_value(state: List[float], depth: int, player: int, model_id: str | None) -> float:
    w = _winner(state)
    if w == 1:
        return 1.0
    if w == -1:
        return -1.0
    if _draw(state):
        return 0.0
    if depth <= 0:
        return _state_value_for_player(state, player=player, model_id=model_id)

    legal = _legal_actions(state)
    if not legal:
        return 0.0

    if player == 1:
        best = float("-inf")
        for a in legal:
            nxt = list(state)
            nxt[a] = 1.0
            best = max(best, _minimax_value(nxt, depth - 1, -1, model_id))
        return best

    best = float("inf")
    for a in legal:
        nxt = list(state)
        nxt[a] = -1.0
        best = min(best, _minimax_value(nxt, depth - 1, 1, model_id))
    return best


def _model_action(
    state: List[float],
    temperature: float,
    model_id: str | None = None,
    search_depth: int = 1,
    player: int = 1,
) -> int:
    legal = _legal_actions(state)
    if not legal:
        return 0
    if search_depth <= 1:
        action, _, _ = _model_action_with_probs(state, temperature=temperature, model_id=model_id, player=player)
        return action
    # Use minimax-style look-ahead with correct current player.
    best_action = legal[0]
    if player == 1:
        best_value = float("-inf")
        for a in legal:
            nxt = list(state)
            nxt[a] = 1.0
            value = _minimax_value(nxt, depth=search_depth - 1, player=-1, model_id=model_id)
            if value > best_value:
                best_value = value
                best_action = a
        return best_action

    best_value = float("inf")
    for a in legal:
        nxt = list(state)
        nxt[a] = -1.0
        value = _minimax_value(nxt, depth=search_depth - 1, player=1, model_id=model_id)
        if value < best_value:
            best_value = value
            best_action = a
    return best_action


def _pick_action(
    policy: str,
    state: List[float],
    temperature: float,
    model_id: str | None = None,
    player: int = 1,
    search_depth: int = 1,
) -> int:
    legal = _legal_actions(state)
    if not legal:
        return 0
    if policy == "random":
        return random.choice(legal)
    return _model_action(state, temperature, model_id=model_id, search_depth=search_depth, player=player)


def _simulate_game(
    x_policy: str,
    o_policy: str,
    x_model_id: str | None,
    o_model_id: str | None,
    temperature: float = 1.0,
    search_depth: int = 1,
) -> tuple[int, int]:
    # Returns (winner, move_count): winner 1 for X, -1 for O, 0 for draw.
    board = [0.0] * 9
    player = 1
    moves = 0
    while True:
        policy = x_policy if player == 1 else o_policy
        model_id = x_model_id if player == 1 else o_model_id
        action = _pick_action(
            policy,
            board,
            temperature=temperature,
            model_id=model_id,
            player=player,
            search_depth=search_depth,
        )
        if board[action] != 0.0:
            return -player, moves
        board[action] = float(player)
        moves += 1
        w = _winner(board)
        if w != 0:
            return w, moves
        if _draw(board):
            return 0, moves
        player *= -1


def _run_training(episodes: int) -> None:
    global active_model_id
    training_status["running"] = True
    training_status["episodes_done"] = 0
    training_status["episodes_total"] = episodes
    best_loss = float("inf")
    best_snapshot: dict | None = None

    def on_episode(item: dict) -> None:
        if stop_event.is_set():
            raise RuntimeError("stopped")
        training_status["episodes_done"] = int(item["episode"])
        nonlocal best_loss, best_snapshot
        loss = float(item.get("loss", 0.0))
        # Track best checkpoint by loss during this training run.
        if loss > 0.0 and loss < best_loss:
            best_loss = loss
            best_snapshot = trainer.snapshot_state()
        metrics_queue.put({"type": "metrics", **item})

    stopped_early = False
    try:
        trainer.train(episodes=episodes, on_episode_end=on_episode)
    except RuntimeError:
        stopped_early = True
    finally:
        should_save = (not stopped_early) or (app.state.stop_behavior == "save")
        model_id = None
        if should_save:
            model_id = datetime.now().strftime("model_%Y%m%d_%H%M%S")
            trainer.save_checkpoint(str(_model_path(model_id)), snapshot=best_snapshot)
            active_model_id = model_id
            model_cache.pop(model_id, None)
        training_status["running"] = False
        stop_event.clear()
        app.state.stop_behavior = "save"
        metrics_queue.put({"type": "training_status", **training_status})
        metrics_queue.put(
            {
                "type": "training_done",
                "saved": should_save,
                "model_id": model_id,
                "stopped_early": stopped_early,
            }
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict:
    return trainer.to_dict() | {"active_model_id": active_model_id}


@app.get("/models/options")
def model_options() -> dict:
    return {
        "policies": [
            {"id": "model", "label": "Saved Model"},
            {"id": "random", "label": "Random Model"},
        ],
        "models": _list_model_entries(),
        "active_model_id": active_model_id,
    }


@app.get("/models/list")
def models_list() -> dict:
    return {"models": _list_model_entries(), "active_model_id": active_model_id}


@app.post("/models/load")
def models_load(req: ModelLoadRequest) -> dict:
    global active_model_id
    p = _model_path(req.model_id)
    if not p.exists():
        return {"ok": False, "message": "model not found"}
    trainer.load_checkpoint(str(p))
    active_model_id = req.model_id
    model_cache.pop(req.model_id, None)
    return {"ok": True, "active_model_id": active_model_id}


@app.post("/models/delete")
def models_delete(req: ModelDeleteRequest) -> dict:
    global active_model_id
    p = _model_path(req.model_id)
    if not p.exists():
        return {"ok": False, "message": "model not found"}
    p.unlink()
    model_cache.pop(req.model_id, None)
    if active_model_id == req.model_id:
        active_model_id = None
    return {"ok": True, "deleted_model_id": req.model_id, "active_model_id": active_model_id}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/training/status")
def train_status() -> dict:
    return dict(training_status)


@app.post("/train/start")
def train_start(req: TrainRequest) -> dict:
    if training_status["running"]:
        return {"ok": False, "message": "training already running", "status": training_status}

    trainer.apply_train_overrides(
        lr=req.lr,
        batch_size=req.batch_size,
        gamma=req.gamma,
        epsilon_start=req.epsilon_start,
        epsilon_end=req.epsilon_end,
        epsilon_decay_steps=req.epsilon_decay_steps,
        target_sync_steps=req.target_sync_steps,
    )
    trainer.reset_for_new_run()

    stop_event.clear()
    app.state.stop_behavior = "save"
    app.state.train_thread = threading.Thread(target=_run_training, args=(req.episodes,), daemon=True)
    app.state.train_thread.start()
    return {"ok": True, "status": training_status}


@app.post("/train/stop")
def train_stop() -> dict:
    if not training_status["running"]:
        return {"ok": True, "message": "training not running", "status": training_status}
    app.state.stop_behavior = "save"
    stop_event.set()
    return {"ok": True, "message": "stop requested, will save best", "status": training_status}


@app.post("/train/skip")
def train_skip() -> dict:
    if not training_status["running"]:
        return {"ok": True, "message": "training not running", "status": training_status}
    app.state.stop_behavior = "skip"
    stop_event.set()
    return {"ok": True, "message": "skip requested, will stop without saving", "status": training_status}


@app.post("/infer")
def infer(req: BoardRequest) -> dict:
    selected_model_id = req.model_id or active_model_id
    temperature = req.tau if req.tau is not None else req.temperature
    legal = [i for i, v in enumerate(req.state) if v == 0.0]

    x_count = sum(1 for v in req.state if v == 1.0)
    o_count = sum(1 for v in req.state if v == -1.0)
    # Valid game states: X starts; when counts equal it's X turn, else O turn.
    current_player = 1 if x_count <= o_count else -1

    if req.policy == "random":
        chosen = random.choice(legal) if legal else 0
        probs = [0.0] * 9
        if legal:
            uniform = 1.0 / len(legal)
            for idx in legal:
                probs[idx] = uniform
        return {
            "q_values": [0.0] * 9,
            "probabilities": probs,
            "top3_moves": [{"action": i, "probability": probs[i], "q_value": 0.0} for i in legal[:3]],
            "chosen_action": chosen,
            "policy": "random",
            "search_depth": req.search_depth,
        }

    _, q_cpu, p_cpu = _model_action_with_probs(
        req.state,
        temperature=temperature,
        model_id=selected_model_id,
        player=current_player,
    )
    ranked = sorted(legal, key=lambda idx: p_cpu[idx], reverse=True)[:3]
    chosen_action = _model_action(
        req.state,
        temperature=temperature,
        model_id=selected_model_id,
        search_depth=req.search_depth,
        player=current_player,
    )
    return {
        "q_values": q_cpu,
        "probabilities": p_cpu,
        "top3_moves": [{"action": i, "probability": p_cpu[i], "q_value": q_cpu[i]} for i in ranked],
        "chosen_action": chosen_action,
        "policy": "model",
        "model_id": selected_model_id,
        "search_depth": req.search_depth,
    }


@app.post("/tournament/run")
def tournament_run(req: TournamentRequest) -> dict:
    x_wins = 0
    o_wins = 0
    draws = 0
    for _ in range(req.games):
        swap = random.random() < 0.5
        x_policy = req.o_policy if swap else req.x_policy
        o_policy = req.x_policy if swap else req.o_policy
        x_model_id = req.o_model_id if swap else req.x_model_id
        o_model_id = req.x_model_id if swap else req.o_model_id
        w, _ = _simulate_game(
            x_policy,
            o_policy,
            x_model_id,
            o_model_id,
            temperature=req.temperature,
            search_depth=req.search_depth,
        )
        if w == 1:
            if swap:
                o_wins += 1
            else:
                x_wins += 1
        elif w == -1:
            if swap:
                x_wins += 1
            else:
                o_wins += 1
        else:
            draws += 1
    return {
        "games": req.games,
        "x_policy": req.x_policy,
        "o_policy": req.o_policy,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "x_win_rate": x_wins / req.games,
        "o_win_rate": o_wins / req.games,
        "x_model_id": req.x_model_id,
        "o_model_id": req.o_model_id,
    }


@app.post("/tournament/leaderboard")
def tournament_leaderboard(req: LeaderboardRequest) -> dict:
    model_ids = [m["id"] for m in _list_model_entries()]
    if active_model_id and active_model_id not in model_ids:
        model_ids.insert(0, active_model_id)

    rows = []
    for mid in model_ids:
        x_wins = 0
        o_wins = 0
        draws = 0
        for _ in range(req.games_per_side):
            w1, _ = _simulate_game("model", "random", mid, None, temperature=1.0, search_depth=1)
            if w1 == 1:
                x_wins += 1
            elif w1 == 0:
                draws += 1

            w2, _ = _simulate_game("random", "model", None, mid, temperature=1.0, search_depth=1)
            if w2 == -1:
                o_wins += 1
            elif w2 == 0:
                draws += 1

        total = req.games_per_side * 2
        score = (x_wins + o_wins + 0.5 * draws) / total
        rows.append(
            {
                "model_id": mid,
                "wins_as_x": x_wins,
                "wins_as_o": o_wins,
                "draws": draws,
                "total_games": total,
                "score": score,
                "win_rate": (x_wins + o_wins) / total,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"games_per_side": req.games_per_side, "leaderboard": rows}


@app.post("/tournament/matrix")
def tournament_matrix(req: MatrixRequest) -> dict:
    model_ids = [m["id"] for m in _list_model_entries()]
    if active_model_id and active_model_id not in model_ids:
        model_ids.insert(0, active_model_id)
    labels = model_ids + ["random"]

    matrix: list[list[dict]] = []
    for row_label in labels:
        row: list[dict] = []
        for col_label in labels:
            if row_label == col_label:
                row.append({"wins": 0, "draws": 0, "losses": 0, "games": 0})
                continue

            x_policy = "random" if row_label == "random" else "model"
            o_policy = "random" if col_label == "random" else "model"
            x_model_id = None if row_label == "random" else row_label
            o_model_id = None if col_label == "random" else col_label

            x_wins = 0
            o_wins = 0
            draws = 0
            for _ in range(req.games):
                w, _ = _simulate_game(x_policy, o_policy, x_model_id, o_model_id, temperature=1.0, search_depth=1)
                if w == 1:
                    x_wins += 1
                elif w == -1:
                    o_wins += 1
                else:
                    draws += 1
            row.append({"wins": x_wins, "draws": draws, "losses": o_wins, "games": req.games})
        matrix.append(row)

    return {"labels": labels, "games_per_pair": req.games, "matrix": matrix}


def _all_tournament_agents() -> list[str]:
    model_ids = [m["id"] for m in _list_model_entries()]
    if active_model_id and active_model_id not in model_ids:
        model_ids.insert(0, active_model_id)
    labels = list(dict.fromkeys(model_ids + ["random"]))
    return labels


@app.websocket("/ws/tournament/all")
async def ws_tournament_all(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        games = int(payload.get("games", 100))
        temperature = float(payload.get("temperature", 1.0))
        search_depth = int(payload.get("search_depth", 1))
        include_model_ids = set(payload.get("include_model_ids", []) or [])
        all_agents = _all_tournament_agents()
        if include_model_ids:
            labels = [lbl for lbl in all_agents if lbl in include_model_ids]
        else:
            labels = list(all_agents)
        if len(labels) < 2:
            await websocket.send_json(
                {
                    "type": "done",
                    "games": 0,
                    "message": "Need at least two agents for tournament.",
                    "leaderboard": [],
                    "labels": labels,
                    "included_model_ids": sorted(include_model_ids),
                    "matrix": [],
                }
            )
            return

        matrix_map: dict[tuple[str, str], dict] = {}
        standings: dict[str, dict] = {
            lbl: {
                "model_id": lbl,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "wins_as_x": 0,
                "wins_as_o": 0,
                "total_games": 0,
            }
            for lbl in labels
        }

        for i in range(1, games + 1):
            x_label, o_label = random.sample(labels, 2)
            x_policy = "random" if x_label == "random" else "model"
            o_policy = "random" if o_label == "random" else "model"
            x_mid = None if x_label == "random" else x_label
            o_mid = None if o_label == "random" else o_label

            winner, moves = _simulate_game(
                x_policy,
                o_policy,
                x_mid,
                o_mid,
                temperature=temperature,
                search_depth=search_depth,
            )
            winner_text = "draw"
            if winner == 1:
                winner_text = "X"
                standings[x_label]["wins"] += 1
                standings[x_label]["wins_as_x"] += 1
                standings[o_label]["losses"] += 1
            elif winner == -1:
                winner_text = "O"
                standings[o_label]["wins"] += 1
                standings[o_label]["wins_as_o"] += 1
                standings[x_label]["losses"] += 1
            else:
                standings[x_label]["draws"] += 1
                standings[o_label]["draws"] += 1

            standings[x_label]["total_games"] += 1
            standings[o_label]["total_games"] += 1

            key = (x_label, o_label)
            if key not in matrix_map:
                matrix_map[key] = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
            matrix_map[key]["games"] += 1
            if winner == 1:
                matrix_map[key]["wins"] += 1
            elif winner == -1:
                matrix_map[key]["losses"] += 1
            else:
                matrix_map[key]["draws"] += 1

            await websocket.send_json(
                {
                    "type": "game_result",
                    "game": i,
                    "games": games,
                    "x_model": x_label,
                    "o_model": o_label,
                    "starter": x_label,
                    "moves": moves,
                    "temperature": temperature,
                    "search_depth": search_depth,
                    "winner": winner_text,
                }
            )

        leaderboard = []
        for lbl, row in standings.items():
            total = max(1, row["total_games"])
            win_rate = row["wins"] / total
            score = (row["wins"] + 0.5 * row["draws"]) / total
            leaderboard.append(
                {
                    "model_id": lbl,
                    "wins_as_x": row["wins_as_x"],
                    "wins_as_o": row["wins_as_o"],
                    "draws": row["draws"],
                    "total_games": row["total_games"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "score": score,
                    "win_rate": win_rate,
                }
            )
        leaderboard.sort(key=lambda r: r["score"], reverse=True)

        matrix: list[list[dict]] = []
        for r in labels:
            row_cells: list[dict] = []
            for c in labels:
                if r == c:
                    row_cells.append({"wins": 0, "draws": 0, "losses": 0, "games": 0})
                else:
                    row_cells.append(matrix_map.get((r, c), {"wins": 0, "draws": 0, "losses": 0, "games": 0}))
            matrix.append(row_cells)

        await websocket.send_json(
            {
                "type": "done",
                "games": games,
                "included_model_ids": sorted(include_model_ids),
                "temperature": temperature,
                "search_depth": search_depth,
                "leaderboard": leaderboard,
                "labels": labels,
                "matrix": matrix,
            }
        )
    except WebSocketDisconnect:
        return


@app.websocket("/ws/tournament")
async def ws_tournament(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        games = int(payload.get("games", 50))
        temperature = float(payload.get("temperature", 1.0))
        search_depth = int(payload.get("search_depth", 1))
        x_model_id = payload.get("x_model_id")
        o_model_id = payload.get("o_model_id")
        x_policy = "model" if x_model_id != "random" else "random"
        o_policy = "model" if o_model_id != "random" else "random"
        x_mid = None if x_model_id in (None, "", "random") else x_model_id
        o_mid = None if o_model_id in (None, "", "random") else o_model_id

        x_wins = 0
        o_wins = 0
        draws = 0
        for i in range(1, games + 1):
            swap = random.random() < 0.5
            game_x_policy = o_policy if swap else x_policy
            game_o_policy = x_policy if swap else o_policy
            game_x_mid = o_mid if swap else x_mid
            game_o_mid = x_mid if swap else o_mid
            game_x_name = (o_model_id or "loaded_model") if swap else (x_model_id or "loaded_model")
            game_o_name = (x_model_id or "loaded_model") if swap else (o_model_id or "loaded_model")
            winner, moves = _simulate_game(
                game_x_policy,
                game_o_policy,
                game_x_mid,
                game_o_mid,
                temperature=temperature,
                search_depth=search_depth,
            )
            winner_text = "draw"
            if winner == 1:
                if swap:
                    o_wins += 1
                else:
                    x_wins += 1
                winner_text = "X"
            elif winner == -1:
                if swap:
                    x_wins += 1
                else:
                    o_wins += 1
                winner_text = "O"
            else:
                draws += 1
            await websocket.send_json(
                {
                    "type": "game_result",
                    "game": i,
                    "games": games,
                    "x_model": game_x_name,
                    "o_model": game_o_name,
                    "starter": game_x_name,
                    "moves": moves,
                    "winner": winner_text,
                }
            )

        await websocket.send_json(
            {
                "type": "done",
                "games": games,
                "x_wins": x_wins,
                "o_wins": o_wins,
                "draws": draws,
                "x_win_rate": x_wins / games if games else 0.0,
            }
        )
    except WebSocketDisconnect:
        return


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await websocket.send_json({"type": "training_status", **training_status})
        while True:
            try:
                payload = await asyncio.to_thread(metrics_queue.get, True, 1.0)
                await websocket.send_json(payload)
            except Empty:
                await websocket.send_json({"type": "heartbeat", "running": training_status["running"]})
    except WebSocketDisconnect:
        return

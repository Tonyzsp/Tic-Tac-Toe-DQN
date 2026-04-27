# Tic-Tac-Toe DQN Teaching Guide

This document explains how this project works, why each component exists, and how to reason about the key parameters.

## 0) Algorithm and Loss (At a Glance)

- Main algorithm: **Deep Q-Network (DQN)** for discrete action control.
- Training strategy: self-play + replay buffer + target network sync.
- Exploration: epsilon-greedy during training.
- Inference policy: masked argmax / temperature policy (and optional depth search).
- Optional decision refinement: minimax-style look-ahead (`search_depth > 1`).

Loss function:
- **Mean Squared Error (MSE)** between predicted Q-value and TD target.
- TD target:
  - Non-terminal: `y = r + gamma * max_a' Q_target(s', a')`
  - Terminal: `y = r`
- Optimized quantity: `MSE(Q_online(s, a), y)`

## 1) What This Project Builds

You have a full local system with:
- A Tic-Tac-Toe game engine
- A DQN (MLP) model and trainer
- A FastAPI backend
- A React frontend for training, play, and tournaments
- Live metrics and an AI Brain visualizer

## 2) Core RL Setup

### State Encoding
- Shape: `(1, 9)`
- Values:
  - `+1.0` current player pieces
  - `-1.0` opponent pieces
  - `0.0` empty

### Network
- MLP:
  - `9 -> 128 -> 64 -> 9`
  - ReLU activations
  - Dropout `0.1`
- Output is one Q-value per board position.

### Objective
- TD learning with MSE loss
- Target network sync for stability
- Replay buffer to reduce temporal correlation

### Rewards
- Win: `+1.0`
- Lose: `-1.0`
- Draw: `+0.5`
- Illegal move: strong penalty and terminate episode

## 3) Illegal Move Safety (Critical)

Before selecting actions, occupied cells are masked as `-inf`.  
This is applied in training and inference so the model does not choose invalid moves.

## 4) Parameters You Actually Feel in UI

### Temperature
- Controls softness of policy probabilities.
- `temperature = 0` means pure greedy (`argmax` over legal moves).
- Higher temperature spreads probability mass more.

### Search Depth
- `1`: use policy directly.
- `>1`: use look-ahead search (minimax-style) to evaluate future responses.

### Why AI Brain and Final Move Can Differ
- AI Brain panel shows network top-3 probabilities.
- Final action may differ when depth search is enabled (`search_depth > 1`), because look-ahead can override the policy top-1.

## 5) Training / Play / Tournament

### Training
- Generates data online by self-play.
- Updates model continuously from replayed transitions.
- Streams train and validation losses in real time.

### Play
- Use saved model or random agent.
- User-vs-AI and AI-vs-AI are both supported.

### Tournament
- Select included models (multi-select).
- Optional `random` agent can be included.
- Stop means no more new matches, then compute ranking and head-to-head matrix from finished games.

## 6) Recommended Local Workflow

1. Backend:
```bash
uvicorn src.api.main:app --reload
```
2. Frontend:
```bash
cd frontend
npm install
npm run dev
```
3. Open dashboard and iterate:
- Train -> save model
- Compare in tournament
- Inspect AI Brain behavior with and without search depth

## 7) Important Parameters (Quick Reference)

Training:
- `episodes`: number of training games in one run.
- `learning_rate` (`lr`): optimizer step size; too high can be unstable.
- `batch_size`: replay samples per update step.
- `gamma`: reward discount factor (how much future reward matters).
- `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`: exploration schedule.
- `target_sync_steps`: how often target network is synchronized.

Inference / Play:
- `temperature`: policy softness (`0` = pure argmax).
- `search_depth`: look-ahead depth (`1` = no extra look-ahead, `>1` = minimax-style search).
- `policy`: `model` or `random`.
- `model_id`: which saved checkpoint to load for inference.

Tournament:
- `games`: total simulated matches.
- `include_model_ids`: selected participants in tournament pool.
- `temperature`, `search_depth`: same semantics as play, but used for tournament agents.

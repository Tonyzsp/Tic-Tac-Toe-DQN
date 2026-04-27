# Tic-Tac-Toe Deep Q-Learning (DQN) Implementation Plan

## 1) Scope and Product Goal

Build a complete Tic-Tac-Toe AI system using Deep Q-Learning with:
- A robust Python game engine
- A reproducible training pipeline
- A PyTorch Lightning DQN module
- A FastAPI service layer with WebSocket streaming
- A React-based interactive dashboard with AI decision transparency

> Assumption: this plan targets **Tic-Tac-Toe**. If the references to "Checkers", "PDN", or "Minimax self-play parser" are intentional for a future extension, they are treated as optional stretch goals outside the core Tic-Tac-Toe milestone.

---

## 2) Technical Specification Baseline

### 2.1 State Representation
- Input tensor shape: `(1, 9)`
- Encoding:
  - `+1.0` = agent piece (X)
  - `-1.0` = opponent piece (O)
  - `0.0` = empty cell
- Symmetry-based augmentation per sample:
  - Rotations: 90, 180, 270 degrees
  - Reflections: horizontal / vertical / diagonal variants
  - Total augmented variants per board: `8`

### 2.2 Network Architecture (MLP)
- Input layer: `9`
- Hidden layer 1: `Linear(9, 128) + ReLU`
- Hidden layer 2: `Linear(128, 64) + ReLU`
- Dropout: `p=0.1`
- Output layer: `Linear(64, 9)` (raw Q-values, linear activation)
- Output semantics: index `i` predicts `Q(s, a_i)`

### 2.3 RL Objective
- TD target:
  - `y = r + gamma * max_a' Q_target(s', a')` for non-terminal transitions
  - `y = r` for terminal transitions
- Loss: MSE between `Q_online(s, a)` and `y`
- Target network sync every `N` optimizer steps

### 2.4 Reward Function
- Win: `+1.0`
- Lose: `-1.0`
- Draw: `+0.5`
- Illegal move: `-10.0` and immediate episode termination

### 2.5 Exploration
- Epsilon-greedy during training:
  - `epsilon` decays linearly from `1.0` to `0.05`
- Inference:
  - `epsilon = 0`
  - action = `argmax_a Q(s, a)`

### 2.6 Action Masking (Critical)
- Before `argmax` or `softmax`, mask occupied cells:
  - set invalid action Q-values to `-inf`
- This must be enforced in:
  - training action selection
  - target estimation (if selecting next action from online net)
  - inference
  - probability visualization in UI

### 2.7 Replay / Optimization
- Replay buffer size: `10,000`
- Batch size: `32` or `64` (configurable)
- Optimizer: Adam, `lr=1e-4`

### 2.8 Policy Probability View
- Convert Q-values to probabilities with temperature softmax:
  - `P(a_i) = exp(Q_i / tau) / sum_j exp(Q_j / tau)`
- Add temperature `tau` as configurable runtime parameter

### 2.9 Look-ahead Evaluation
- Optional evaluation-only 1-ply look-ahead:
  - Agent chooses candidate action
  - Simulate opponent best response (adversarial assumption)
  - Re-score candidate action with opponent response penalty
- Keep this separate from core DQN training loop to avoid instability.

---

## 3) System Architecture

### Backend Modules
- `engine/`: board state, move validation, terminal detection, reward emission
- `rl/`: replay buffer, epsilon schedule, training loop, evaluation scripts
- `model/`: Lightning module, MLP network, target sync logic
- `api/`: FastAPI endpoints, game session API, WebSocket metric stream

### Frontend Modules
- `dashboard/`: training controls and charts
- `board/`: interactive Tic-Tac-Toe board
- `brain-panel/`: top-3 moves + probability display during AI turn

---

## 4) Execution Phases

## Phase 1: Game Engine
**Objective:** implement deterministic Tic-Tac-Toe core logic.

Deliverables:
- Board representation (`9` cells or `3x3` view adapter)
- Move validation API
- Win/draw/loss detection
- Illegal move handling with terminal penalty
- Unit tests for all terminal and invalid-action edge cases

Exit criteria:
- 100% pass on predefined engine tests
- Illegal moves always rejected and flagged terminal when requested by training mode

## Phase 2: Data and Self-Play Pipeline
**Objective:** generate high-quality transitions for DQN.

Deliverables:
- Self-play runner (epsilon-greedy agent vs configurable opponent)
- Symmetry augmentation utility (8 transforms + inverse mapping if needed)
- Replay buffer persistence (optional checkpoint save/load)
- Dataset/statistics export for diagnostics

Exit criteria:
- Stable generation of transitions without invalid index/state errors
- Augmented samples validated for value/action consistency

## Phase 3: DQN Model (PyTorch Lightning)
**Objective:** implement trainable, configurable DQN module.

Deliverables:
- MLP model per spec
- Online/target network update logic
- TD-loss computation with terminal masking
- Dynamic hyperparameter injection (CLI/env/config)
- Checkpointing and resume support

Exit criteria:
- Training loss decreases over baseline runs
- No illegal action selected when masking is enabled

## Phase 4: API Layer (FastAPI + WebSockets)
**Objective:** expose training and gameplay services.

Deliverables:
- REST endpoints:
  - start/stop training
  - get current config and model status
  - run inference for a board state
- WebSocket stream for real-time metrics:
  - training loss
  - validation loss
  - win/draw/loss rates
  - illegal move rate

Exit criteria:
- Frontend can subscribe and render live metrics with reconnect handling

## Phase 5: Frontend UI (React Dashboard)
**Objective:** provide interactive control and model interpretability.

Deliverables:
- **Configuration Panel**
  - epsilon schedule
  - learning rate
  - batch size
  - gamma
  - target sync interval
  - temperature (`tau`)
  - search depth (for optional look-ahead evaluator)
- **Live Metrics**
  - real-time line charts for training loss vs validation loss
  - win rate and illegal move rate indicators
- **Interactive Game Board**
  - play against current model
  - legal-cell highlighting
- **AI Brain Visualizer**
  - top 3 candidate moves
  - Q-values and softmax probabilities
  - currently applied action mask status

Exit criteria:
- User can train, monitor, and play from one UI
- AI decision panel updates every AI turn without blocking gameplay

---

## 5) Evaluation and Acceptance Metrics

Primary success metrics:
- Random-opponent win rate `> 95%`
- Illegal move rate reaches `0%` after ~500 episodes
- Q-value trend for strong openings (e.g., center cell) stabilizes upward

Secondary quality metrics:
- Training throughput (episodes/sec)
- API latency for single-state inference
- Frontend update smoothness under live metric streaming

---

## 6) Risks and Mitigations

- **Risk:** Q-learning instability in self-play  
  **Mitigation:** target network sync, replay randomization, conservative LR

- **Risk:** illegal actions leaking into inference  
  **Mitigation:** centralized action-mask utility reused in all code paths

- **Risk:** misleading probability visualization  
  **Mitigation:** compute probabilities only after legal-action masking

- **Risk:** frontend metric lag or disconnects  
  **Mitigation:** WebSocket heartbeat, reconnect, and last-value cache

---

## 7) Definition of Done

Project is complete when:
- All five phases pass exit criteria
- Metrics meet primary thresholds
- End-to-end demo shows:
  - training start/stop
  - live chart updates
  - interactive gameplay
  - transparent top-3 AI move reasoning panel

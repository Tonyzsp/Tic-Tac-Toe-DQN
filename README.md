# Tic-Tac-Toe-DQN

Deep Q-Learning Tic-Tac-Toe project with:
- Python backend (FastAPI + PyTorch)
- React frontend (Vite + Chart.js)
- Real-time training metrics and tournament dashboard

For a complete learning guide, see `teach.md`.

## Architecture (Frontend + Backend)

- Backend API default: `http://127.0.0.1:8000`
- Frontend dev server default: `http://127.0.0.1:5173`
- Frontend reads `VITE_API_BASE` and defaults to `http://127.0.0.1:8000`
- You should run backend and frontend in separate terminals.

## Prerequisites

- Python 3.10+ recommended
- Node.js 18+ recommended
- npm

## Backend Setup

From project root:

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn torch websockets
```

Start backend:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Health check:
- Open `http://127.0.0.1:8000/health`

Optional quick CLI training test:

```bash
python scripts/train.py
```

## Frontend Setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:
- `http://127.0.0.1:5173`

If backend is not on default URL, run:

```bash
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run dev
```

## Run Flow (Recommended)

1. Start backend first.
2. Start frontend second.
3. Open frontend dashboard.
4. Use Training tab to start training and save models.
5. Use Play and Tournament tabs to evaluate behavior.

## API Endpoints

- `GET /health`
- `GET /config`
- `GET /models/options`
- `GET /models/list`
- `POST /models/load` with body: `{"model_id": "model_YYYYMMDD_HHMMSS"}`
- `GET /training/status`
- `POST /train/start` with body: `{"episodes": 100}`
- `POST /train/stop`
- `POST /infer` with body:
  - `state`: length-9 board values (`+1`, `-1`, `0`)
  - `temperature`: policy temperature (`0` means pure argmax)
  - `search_depth`: look-ahead depth (minimax-style evaluation)
  - `policy`: `model` or `random`
  - `model_id`: optional saved model id (for model policy)
  - `tau`: legacy alias (still supported)
- `POST /tournament/run` with body:
  - `games`, `x_policy`, `o_policy`
  - optional `x_model_id`, `o_model_id`
- `WS /ws/metrics` for real-time episode metrics

## Project Structure

- `src/engine/tictactoe.py`: game rules, reward, illegal action handling
- `src/model/dqn.py`: MLP DQN model, action masking, softmax policy
- `src/rl/buffer.py`: replay buffer
- `src/rl/trainer.py`: self-play training loop with target network
- `src/api/main.py`: FastAPI service layer

## Important Behavior

Illegal moves are blocked by mask logic before action selection:
- occupied board positions are assigned `-inf` for argmax/softmax
- this is applied in both training and inference paths

AI Brain Visualizer note:
- The panel shows top-3 probabilities from the network policy.
- The final chosen move can differ when `search_depth > 1`, because look-ahead search may override the network's top probability choice.

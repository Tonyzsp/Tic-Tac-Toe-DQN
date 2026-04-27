# Tic-Tac-Toe-DQN

Deep Q-Learning implementation for Tic-Tac-Toe using PyTorch.  
Includes self-play training, epsilon-greedy exploration, legal-action masking, and API inference with top-3 move probabilities.

## Quick Start

```bash
pip install -r requirements.txt
```

Run a short training loop:

```bash
python scripts/train.py
```

Start API server:

```bash
uvicorn src.api.main:app --reload
```

## Separate Frontend (Vite + React)

Run frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open frontend dashboard:

- `http://127.0.0.1:5173/`

Optional API override:

```bash
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

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
  - `temperature`: softmax temperature
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

## Windows Torch DLL Note

If you hit `WinError 1114` while importing torch, it is usually an environment dependency issue (CUDA/VC runtime/conda DLL conflict).  
Try a clean virtual environment and reinstall CPU-only torch first:

```bash
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

# Tic-Tac-Toe-DQN

Deep Q-Learning Tic-Tac-Toe project with:
- Python backend (FastAPI + PyTorch)
- React frontend (Vite + Chart.js)
- Real-time training metrics and tournament dashboard

For more tech info, see `tech.md`.

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

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Project Structure

```text
Tic-Tac-Toe-DQN/
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # main dashboard UI and interactions
│   │   ├── App.css                  # dashboard styles
│   │   ├── main.jsx                 # React entry
│   │   └── index.css                # global styles
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── train.py                     # quick CLI training script
├── src/
│   ├── api/
│   │   └── main.py                  # FastAPI endpoints and WebSocket streams
│   ├── engine/
│   │   └── tictactoe.py             # game rules, terminal checks, rewards
│   ├── model/
│   │   └── dqn.py                   # DQN MLP, action masking utilities
│   └── rl/
│       ├── buffer.py                # replay buffer
│       └── trainer.py               # self-play training loop + target sync
├── static/
│   └── index.html
├── models/                          # local saved checkpoints (not committed)
├── requirements.txt
├── tech.md
├── README.md
└── .gitignore
```

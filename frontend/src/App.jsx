import { useEffect, useMemo, useRef, useState } from 'react'
import Chart from 'chart.js/auto'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const WIN_LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

function movingAverage(values, windowSize = 15) {
  if (!Array.isArray(values) || values.length === 0) return []
  const out = []
  for (let i = 0; i < values.length; i += 1) {
    const start = Math.max(0, i - windowSize + 1)
    const slice = values.slice(start, i + 1)
    const avg = slice.reduce((a, b) => a + b, 0) / slice.length
    out.push(avg)
  }
  return out
}

function getGameResult(board) {
  for (const [a, b, c] of WIN_LINES) {
    const s = board[a] + board[b] + board[c]
    if (s === 3) return 'x_win'
    if (s === -3) return 'o_win'
  }
  return board.every((v) => v !== 0) ? 'draw' : null
}

function SliderField({ label, value, min, max, step, onChange, helpText, formatValue }) {
  const shown = formatValue ? formatValue(value) : value
  return (
    <>
      <label title={helpText || ''}>{label} <span className="sliderVal">{shown}</span></label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </>
  )
}

function App() {
  const [tab, setTab] = useState('play')
  const [board, setBoard] = useState(Array(9).fill(0))
  const [turn, setTurn] = useState(1)
  const [gameResult, setGameResult] = useState(null)
  const [gameMode, setGameMode] = useState('user_vs_ai')
  const [userSide, setUserSide] = useState('X')
  const [gameStarted, setGameStarted] = useState(false)
  const [xPlayModelId, setXPlayModelId] = useState('random')
  const [oPlayModelId, setOPlayModelId] = useState('random')
  const [autoPlay, setAutoPlay] = useState(false)

  const [episodes, setEpisodes] = useState(2000)
  const [learningRate, setLearningRate] = useState(1e-4)
  const [batchSize, setBatchSize] = useState(64)
  const [gamma, setGamma] = useState(0.99)
  const [epsilonStart, setEpsilonStart] = useState(1.0)
  const [epsilonEnd, setEpsilonEnd] = useState(0.05)
  const [epsilonDecaySteps, setEpsilonDecaySteps] = useState(10000)
  const [targetSyncSteps, setTargetSyncSteps] = useState(200)
  const [temperature, setTemperature] = useState(1.0)
  const [searchDepth, setSearchDepth] = useState(2)
  const [activeModelId, setActiveModelId] = useState('')
  const [status, setStatus] = useState({ running: false, episodes_done: 0, episodes_total: 0 })
  const [metrics, setMetrics] = useState([])
  const [lastMetric, setLastMetric] = useState(null)
  const [uiMessage, setUiMessage] = useState('Idle')
  const [toast, setToast] = useState('')
  const [wsConnected, setWsConnected] = useState(false)
  const [brain, setBrain] = useState([])
  const [aiProbabilities, setAiProbabilities] = useState(Array(9).fill(0))
  const [policyOptions, setPolicyOptions] = useState([{ id: 'model', label: 'Saved Model' }, { id: 'random', label: 'Random Model' }])
  const [modelOptions, setModelOptions] = useState([])

  const [tournamentGames, setTournamentGames] = useState(50)
  const [tournamentTemperature, setTournamentTemperature] = useState(1.0)
  const [tournamentSearchDepth, setTournamentSearchDepth] = useState(2)
  const [tournamentIncludedModels, setTournamentIncludedModels] = useState([])
  const [tournamentResult, setTournamentResult] = useState(null)
  const [tournamentLogs, setTournamentLogs] = useState([])
  const [tournamentRunning, setTournamentRunning] = useState(false)
  const [leaderboard, setLeaderboard] = useState([])
  const [matrixLabels, setMatrixLabels] = useState([])
  const [matrixData, setMatrixData] = useState([])

  const chartRef = useRef(null)
  const chartObj = useRef(null)
  const toastTimerRef = useRef(null)
  const tournamentWsRef = useRef(null)
  const symbols = useMemo(() => board.map((v) => (v === 1 ? 'X' : v === -1 ? 'O' : '')), [board])
  const userMarker = userSide === 'X' ? 1 : -1
  const aiMarker = -userMarker
  const selectedModelSummary = useMemo(
    () => modelOptions.find((m) => m.id === activeModelId) || null,
    [modelOptions, activeModelId],
  )

  const clearLiveMetrics = () => {
    setMetrics([])
    setLastMetric(null)
    if (chartObj.current) {
      chartObj.current.data.labels = []
      chartObj.current.data.datasets.forEach((ds) => { ds.data = [] })
      chartObj.current.update()
    }
  }

  const showToast = (msg) => {
    setToast(msg)
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
    toastTimerRef.current = setTimeout(() => setToast(''), 2600)
  }

  const refreshModelOptions = async () => {
    const d = await fetch(`${API_BASE}/models/options`).then((r) => r.json())
    if (d.policies) setPolicyOptions(d.policies)
    if (d.models) setModelOptions(d.models)
    if (d.active_model_id) {
      setActiveModelId(d.active_model_id)
      setXPlayModelId(d.active_model_id)
      setOPlayModelId(d.active_model_id)
    }
  }

  useEffect(() => {
    setTournamentIncludedModels((prev) => {
      const available = modelOptions.map((m) => m.id)
      if (prev.length === 0) return ['random', ...available]
      return prev.filter((id) => id === 'random' || available.includes(id))
    })
  }, [modelOptions])

  useEffect(() => {
    fetch(`${API_BASE}/training/status`).then((r) => r.json()).then(setStatus).catch(() => {})
    refreshModelOptions().catch(() => {})
    const ws = new WebSocket(`${API_BASE.replace('http://', 'ws://').replace('https://', 'wss://')}/ws/metrics`)
    ws.onopen = () => { setWsConnected(true); setUiMessage('Connected to metrics stream.') }
    ws.onclose = () => { setWsConnected(false); setUiMessage('Metrics stream disconnected.') }
    ws.onmessage = (evt) => {
      const d = JSON.parse(evt.data)
      if (d.type === 'metrics') { setMetrics((p) => [...p.slice(-99), d]); setLastMetric(d) }
      if (d.type === 'training_status') setStatus(d)
      if (d.type === 'training_done') {
        if (d.saved) {
          setUiMessage(`Training finished. Saved model: ${d.model_id}`)
          showToast(`Training finished. Saved model: ${d.model_id}`)
        } else {
          setUiMessage('Training skipped without saving model.')
          showToast('Training stopped/skipped. Model not saved.')
        }
        refreshModelOptions().catch(() => {})
      }
    }
    return () => ws.close()
  }, [])

  useEffect(() => {
    if (!chartRef.current || metrics.length === 0) return
    const labels = metrics.map((m) => m.episode)
    const loss = metrics.map((m) => Number(m.loss ?? 0))
    const valLoss = metrics.map((m) => Number(m.val_loss ?? m.loss ?? 0))
    const lossSmooth = movingAverage(loss, 20)
    const valLossSmooth = movingAverage(valLoss, 20)
    if (!chartObj.current) {
      chartObj.current = new Chart(chartRef.current, {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'Training Loss (raw)', data: loss, borderColor: 'rgba(110,168,254,0.25)', pointRadius: 0, tension: 0.15 },
            { label: 'Validation Loss (raw)', data: valLoss, borderColor: 'rgba(52,211,153,0.25)', pointRadius: 0, tension: 0.15 },
            { label: 'Training Loss (smoothed)', data: lossSmooth, borderColor: '#6ea8fe', borderWidth: 2.5, pointRadius: 0, tension: 0.25 },
            { label: 'Validation Loss (smoothed)', data: valLossSmooth, borderColor: '#34d399', borderWidth: 2.5, pointRadius: 0, tension: 0.25 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#9ca3af' } } },
          scales: {
            x: { ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } },
            y: { ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } },
          },
        },
      })
    } else {
      chartObj.current.data.labels = labels
      chartObj.current.data.datasets[0].data = loss
      chartObj.current.data.datasets[1].data = valLoss
      chartObj.current.data.datasets[2].data = lossSmooth
      chartObj.current.data.datasets[3].data = valLossSmooth
      chartObj.current.update()
    }
  }, [metrics])

  const callInfer = async (state, inferPolicy, inferModelId) => {
    const data = await fetch(`${API_BASE}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state, temperature, policy: inferPolicy ?? 'model', model_id: inferModelId || null, search_depth: searchDepth }),
    }).then((r) => r.json())
    setBrain(data.top3_moves || [])
    setAiProbabilities(Array.isArray(data.probabilities) ? data.probabilities : Array(9).fill(0))
    return data
  }

  const playAiTurn = async () => {
    if (gameResult) return
    const agent = turn === 1 ? xPlayModelId : oPlayModelId
    const p = agent === 'random' ? 'random' : 'model'
    const m = agent === 'random' ? null : (agent || null)
    const d = await callInfer(board, p, m)
    const a = d.chosen_action ?? d.top3_moves?.[0]?.action
    if (a === undefined || board[a] !== 0) return
    const n = [...board]; n[a] = turn; setBoard(n)
    const r = getGameResult(n)
    if (r) setGameResult(r); else setTurn((t) => -t)
  }

  useEffect(() => {
    if (!autoPlay || gameMode !== 'ai_vs_ai' || gameResult) return
    const id = setTimeout(() => { playAiTurn().catch(() => {}) }, 250)
    return () => clearTimeout(id)
  }, [autoPlay, gameMode, gameResult, board, turn])

  useEffect(() => {
    resetBoard()
  }, [gameMode, userSide])

  useEffect(() => {
    if (gameMode !== 'user_vs_ai' || !gameStarted || gameResult) return
    if (turn !== userMarker) {
      const id = setTimeout(() => { playAiTurn().catch(() => {}) }, 250)
      return () => clearTimeout(id)
    }
  }, [gameMode, gameStarted, gameResult, board, turn, userMarker])

  const playHuman = async (idx) => {
    if (gameMode !== 'user_vs_ai' || !gameStarted || gameResult || board[idx] !== 0 || turn !== userMarker) return
    const b1 = [...board]
    b1[idx] = turn
    setBoard(b1)
    const r = getGameResult(b1)
    if (r) setGameResult(r); else setTurn(-turn)
  }

  const resetBoard = () => {
    setBoard(Array(9).fill(0))
    setTurn(gameMode === 'user_vs_ai' ? 1 : (Math.random() < 0.5 ? 1 : -1))
    setGameResult(null)
    setBrain([])
    setAiProbabilities(Array(9).fill(0))
    setAutoPlay(false)
    setGameStarted(gameMode !== 'user_vs_ai')
  }

  const startUserGame = () => {
    setBoard(Array(9).fill(0))
    setTurn(1)
    setGameResult(null)
    setBrain([])
    setAiProbabilities(Array(9).fill(0))
    setAutoPlay(false)
    setGameStarted(true)
  }
  const startTraining = async () => {
    clearLiveMetrics()
    setStatus((s) => ({ ...s, running: true, episodes_done: 0, episodes_total: episodes }))
    setUiMessage('Starting training...')
    const d = await fetch(`${API_BASE}/train/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episodes,
        lr: learningRate,
        batch_size: batchSize,
        gamma,
        epsilon_start: epsilonStart,
        epsilon_end: epsilonEnd,
        epsilon_decay_steps: epsilonDecaySteps,
        target_sync_steps: targetSyncSteps,
      }),
    }).then((r) => r.json())
    if (d.ok) {
      setUiMessage('Training started.')
      return
    }
    setStatus((s) => ({ ...s, running: false }))
    setUiMessage(d.message || 'Failed')
  }
  const stopTraining = async () => { const d = await fetch(`${API_BASE}/train/stop`, { method: 'POST' }).then((r) => r.json()); setUiMessage(d.message || 'Stop') }
  const skipTraining = async () => { const d = await fetch(`${API_BASE}/train/skip`, { method: 'POST' }).then((r) => r.json()); setUiMessage(d.message || 'Skip') }
  const loadModel = async (modelId) => {
    if (!modelId) return
    const d = await fetch(`${API_BASE}/models/load`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: modelId }) }).then((r) => r.json())
    if (d.ok) {
      setActiveModelId(d.active_model_id)
      setUiMessage(`Loaded model: ${d.active_model_id}`)
    } else {
      setUiMessage(d.message || 'Load failed')
    }
  }
  const deleteModel = async (modelId) => {
    if (!modelId) {
      setUiMessage('Please select a model to delete.')
      return
    }
    const d = await fetch(`${API_BASE}/models/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: modelId }) }).then((r) => r.json())
    if (!d.ok) {
      setUiMessage(d.message || 'Delete failed')
      return
    }
    setUiMessage(`Deleted model: ${d.deleted_model_id}`)
    if (d.active_model_id !== undefined) setActiveModelId(d.active_model_id || '')
    await refreshModelOptions()
  }
  const runTournament = async () => {
    if (tournamentRunning) return
    const uniqueIncluded = Array.from(new Set(tournamentIncludedModels))
    if (uniqueIncluded.length < 2) {
      setUiMessage('Tournament needs at least 2 selected models.')
      showToast('Please select at least 2 models in Include Models.')
      return
    }
    setTournamentResult(null)
    setLeaderboard([])
    setMatrixLabels([])
    setMatrixData([])
    setTournamentLogs([])
    setTournamentRunning(true)
    const ws = new WebSocket(`${API_BASE.replace('http://', 'ws://').replace('https://', 'wss://')}/ws/tournament/all`)
    tournamentWsRef.current = ws
    ws.onopen = () => {
      ws.send(JSON.stringify({ games: tournamentGames, temperature: tournamentTemperature, search_depth: tournamentSearchDepth, include_model_ids: uniqueIncluded }))
    }
    ws.onmessage = (evt) => {
      const d = JSON.parse(evt.data)
      if (d.type === 'game_result') {
        setTournamentLogs((prev) => [...prev.slice(-199), d])
      }
      if (d.type === 'done') {
        setTournamentResult(d)
        setLeaderboard(d.leaderboard || [])
        setMatrixLabels(d.labels || [])
        setMatrixData(d.matrix || [])
        setTournamentRunning(false)
        tournamentWsRef.current = null
        ws.close()
      }
    }
    ws.onclose = () => {
      setTournamentRunning(false)
      tournamentWsRef.current = null
    }
  }
  const buildTournamentResultsFromLogs = (logs) => {
    const labels = Array.from(new Set(tournamentIncludedModels))
    if (labels.length < 2) return null

    const standings = Object.fromEntries(labels.map((lbl) => [lbl, { model_id: lbl, wins: 0, losses: 0, draws: 0, wins_as_x: 0, wins_as_o: 0, total_games: 0 }]))
    const pairMap = {}
    for (const g of logs) {
      const x = g.x_model
      const o = g.o_model
      if (!standings[x] || !standings[o]) continue
      standings[x].total_games += 1
      standings[o].total_games += 1
      const key = `${x}__${o}`
      if (!pairMap[key]) pairMap[key] = { wins: 0, draws: 0, losses: 0, games: 0 }
      pairMap[key].games += 1
      if (g.winner === 'X') {
        standings[x].wins += 1
        standings[x].wins_as_x += 1
        standings[o].losses += 1
        pairMap[key].wins += 1
      } else if (g.winner === 'O') {
        standings[o].wins += 1
        standings[o].wins_as_o += 1
        standings[x].losses += 1
        pairMap[key].losses += 1
      } else {
        standings[x].draws += 1
        standings[o].draws += 1
        pairMap[key].draws += 1
      }
    }

    const leaderboard = Object.values(standings).map((r) => {
      const total = Math.max(1, r.total_games)
      return { ...r, score: (r.wins + 0.5 * r.draws) / total, win_rate: r.wins / total }
    }).sort((a, b) => b.score - a.score)

    const matrix = labels.map((row) => labels.map((col) => {
      if (row === col) return { wins: 0, draws: 0, losses: 0, games: 0 }
      return pairMap[`${row}__${col}`] || { wins: 0, draws: 0, losses: 0, games: 0 }
    }))

    return { labels, leaderboard, matrix }
  }
  const stopTournament = () => {
    if (!tournamentRunning) return
    if (tournamentWsRef.current) {
      tournamentWsRef.current.close()
      tournamentWsRef.current = null
    }
    setTournamentRunning(false)
    const partial = buildTournamentResultsFromLogs(tournamentLogs)
    if (partial) {
      setLeaderboard(partial.leaderboard)
      setMatrixLabels(partial.labels)
      setMatrixData(partial.matrix)
      setTournamentResult({ games: tournamentLogs.length })
    }
    setUiMessage('Tournament stopped.')
    showToast(`Tournament stopped. Used ${tournamentLogs.length} finished matches.`)
  }
  const gameResultText = (() => {
    if (gameMode === 'user_vs_ai') {
      if (gameResult === 'draw') return 'Draw'
      if (gameResult === 'x_win') return userMarker === 1 ? 'User wins' : 'AI wins'
      if (gameResult === 'o_win') return userMarker === -1 ? 'User wins' : 'AI wins'
      return ''
    }
    return gameResult === 'x_win' ? 'X wins' : gameResult === 'o_win' ? 'O wins' : gameResult === 'draw' ? 'Draw' : ''
  })()
  const agentLabel = (marker) => {
    const side = marker === 1 ? 'X' : 'O'
    if (gameMode === 'user_vs_ai') {
      if (marker === userMarker) return `User (${side})`
      const modelId = marker === 1 ? xPlayModelId : oPlayModelId
      return `AI (${side}${modelId ? `: ${modelId}` : ''})`
    }
    const modelId = marker === 1 ? xPlayModelId : oPlayModelId
    return `AI ${side}${modelId ? ` (${modelId})` : ''}`
  }
  const isAiTurn = !gameResult && (
    gameMode === 'ai_vs_ai' ||
    (gameMode === 'user_vs_ai' && gameStarted && turn !== userMarker)
  )
  const playStatusText = gameResult ? `Game Over: ${gameResultText}` : `In Progress: ${agentLabel(turn)} to move`

  return (
    <div className="wrap">
      <div className="topbar">
        <div className="brandBlock">
          <h2>Tic-Tac-Toe DQN</h2>
          <p className="brandSub">Interactive train-play-tournament dashboard</p>
        </div>
        <div className="tabs">
          <button className={tab === 'play' ? 'tab active' : 'tab'} onClick={() => setTab('play')}>Play</button>
          <button className={tab === 'training' ? 'tab active' : 'tab'} onClick={() => setTab('training')}>Training</button>
          <button className={tab === 'tournament' ? 'tab active' : 'tab'} onClick={() => setTab('tournament')}>Tournament</button>
        </div>
      </div>
      <div className="appShell">
        <aside className="sidebar card">
          {tab === 'play' && <>
            <h3>Play Parameters</h3>
            <div className="controls vertical">
              <label title="Play mode: human vs AI, or AI vs AI.">Mode</label><select value={gameMode} onChange={(e) => setGameMode(e.target.value)}><option value="user_vs_ai">User vs AI</option><option value="ai_vs_ai">AI vs AI</option></select>
              {gameMode === 'user_vs_ai' && <><label title="Choose your preferred side. First player is still randomized each new game.">User Side Preference</label><select value={userSide} onChange={(e) => setUserSide(e.target.value)}><option value="X">X</option><option value="O">O</option></select></>}
              {(gameMode === 'ai_vs_ai' || userSide === 'O') && (
                <>
                  <label title="Agent used when X moves.">X Agent</label>
                  <select value={xPlayModelId} onChange={(e) => setXPlayModelId(e.target.value)}>
                    <option value="random">Random Model</option>
                    {modelOptions.map((m) => <option key={`xm-${m.id}`} value={m.id}>{m.id}</option>)}
                  </select>
                </>
              )}
              {(gameMode === 'ai_vs_ai' || userSide === 'X') && (
                <>
                  <label title="Agent used when O moves.">O Agent</label>
                  <select value={oPlayModelId} onChange={(e) => setOPlayModelId(e.target.value)}>
                    <option value="random">Random Model</option>
                    {modelOptions.map((m) => <option key={`om-${m.id}`} value={m.id}>{m.id}</option>)}
                  </select>
                </>
              )}
              {gameMode === 'user_vs_ai' && (
                <p className="muted">User side: {userSide} | AI side: {userSide === 'X' ? 'O' : 'X'}</p>
              )}
              <SliderField label="Temperature" value={temperature} min={0} max={2.0} step={0.1} onChange={setTemperature} helpText="Controls randomness in move probabilities. 0 = pure argmax, higher = more exploratory." formatValue={(v) => v.toFixed(2)} />
              <SliderField label="Search Depth" value={searchDepth} min={1} max={6} step={1} onChange={setSearchDepth} helpText="How many plies to look ahead in minimax-style evaluation." />
            </div>
            <div className="actions">
              {gameMode === 'user_vs_ai' && <button onClick={startUserGame}>Start Game</button>}
              <button onClick={resetBoard}>Reset</button>
              {gameMode === 'ai_vs_ai' && <button onClick={() => playAiTurn()}>Next Turn</button>}
              {gameMode === 'ai_vs_ai' && <button onClick={() => setAutoPlay((v) => !v)}>{autoPlay ? 'Stop Auto' : 'Auto Play'}</button>}
            </div>
            <p className="muted">Turn: {turn === 1 ? 'X' : 'O'} | Loaded model: {activeModelId || '(none)'}</p>
          </>}

          {tab === 'training' && <>
            <div className="subBlock">
              <h3>Training Parameters</h3>
              <div className="controls vertical">
                <SliderField label="Episodes" value={episodes} min={500} max={20000} step={100} onChange={setEpisodes} helpText="Number of self-play episodes in this training run." />
                <SliderField label="Learning Rate" value={learningRate} min={0.0001} max={0.005} step={0.0001} onChange={setLearningRate} helpText="Optimizer step size. Too high may oscillate, too low may learn slowly." formatValue={(v) => v.toFixed(4)} />
                <SliderField label="Batch Size" value={batchSize} min={16} max={256} step={16} onChange={setBatchSize} helpText="How many replay samples per gradient step." />
                <SliderField label="Gamma" value={gamma} min={0.8} max={1.0} step={0.01} onChange={setGamma} helpText="Discount factor for future rewards." formatValue={(v) => v.toFixed(2)} />
                <SliderField label="Epsilon Start" value={epsilonStart} min={0.5} max={1.0} step={0.01} onChange={setEpsilonStart} helpText="Initial exploration rate." formatValue={(v) => v.toFixed(2)} />
                <SliderField label="Epsilon End" value={epsilonEnd} min={0.01} max={0.3} step={0.01} onChange={setEpsilonEnd} helpText="Final exploration rate after decay." formatValue={(v) => v.toFixed(2)} />
                <SliderField label="Epsilon Decay Steps" value={epsilonDecaySteps} min={500} max={20000} step={500} onChange={setEpsilonDecaySteps} helpText="How quickly epsilon decays from start to end." />
                <SliderField label="Target Sync Steps" value={targetSyncSteps} min={20} max={1000} step={20} onChange={setTargetSyncSteps} helpText="How often to copy online net weights to target net." />
              </div>
            </div>
            <div className="actions"><button onClick={startTraining} disabled={status.running}>Start</button><button onClick={stopTraining} disabled={!status.running}>Stop & Save Best</button><button onClick={skipTraining} disabled={!status.running}>Skip (No Save)</button></div>
            <div className="subBlock">
              <h3>Model Management</h3>
              <div className="controls vertical">
                <label title="Load one saved model as the currently loaded model.">Load Model</label>
                <select value={activeModelId || ''} onChange={(e) => loadModel(e.target.value)}>
                  <option value="">select to load...</option>
                  {modelOptions.map((m) => <option key={`lm-${m.id}`} value={m.id}>{m.id}</option>)}
                </select>
                <button type="button" onClick={() => deleteModel(activeModelId)} disabled={!activeModelId}>Delete Model</button>
                <div className="modelSummaryBox">
                  <div className="muted">Loaded: {activeModelId || '(none)'}</div>
                  <div className="muted">Step: {selectedModelSummary?.global_step ?? '-'}</div>
                  <div className="muted">Train loss: {selectedModelSummary?.last_loss != null ? Number(selectedModelSummary.last_loss).toFixed(4) : '-'}</div>
                  <div className="muted">Val loss: {selectedModelSummary?.last_val_loss != null ? Number(selectedModelSummary.last_val_loss).toFixed(4) : '-'}</div>
                </div>
              </div>
            </div>
            <p className="muted">Running: {String(status.running)} | Ep: {status.episodes_done}/{status.episodes_total}</p>
            <p className="muted">WS: {wsConnected ? 'connected' : 'disconnected'} | {uiMessage}</p>
          </>}

          {tab === 'tournament' && <>
            <h3>Tournament Parameters</h3>
            <div className="controls vertical">
              <SliderField label="Total Games" value={tournamentGames} min={20} max={1000} step={10} onChange={setTournamentGames} helpText="Randomly match all agents (saved models + random) for this many games." />
              <SliderField label="Tournament Temperature" value={tournamentTemperature} min={0} max={2.0} step={0.1} onChange={setTournamentTemperature} helpText="Softmax temperature used by model agents in tournament matches. 0 = pure argmax." formatValue={(v) => v.toFixed(2)} />
              <SliderField label="Tournament Search Depth" value={tournamentSearchDepth} min={1} max={6} step={1} onChange={setTournamentSearchDepth} helpText="Look-ahead depth used by model agents in tournament matches." />
              <label title="Include only selected models in this tournament run.">Include Models</label>
              <div className="includeModelList">
                <label key="inc-random" className="includeItem">
                  <span title="random">random</span>
                  <input
                    type="checkbox"
                    checked={tournamentIncludedModels.includes('random')}
                    onChange={(e) => {
                      setTournamentIncludedModels((prev) => e.target.checked ? [...prev, 'random'] : prev.filter((id) => id !== 'random'))
                    }}
                  />
                </label>
                {modelOptions.map((m) => {
                  const checked = tournamentIncludedModels.includes(m.id)
                  return (
                    <label key={`inc-${m.id}`} className="includeItem">
                      <span title={m.id}>{m.id}</span>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          setTournamentIncludedModels((prev) => e.target.checked ? [...prev, m.id] : prev.filter((id) => id !== m.id))
                        }}
                      />
                    </label>
                  )
                })}
              </div>
              <p className="muted">Selected: {Array.from(new Set(tournamentIncludedModels)).length} (need at least 2)</p>
            </div>
            <div className="actions"><button onClick={runTournament} disabled={tournamentRunning || Array.from(new Set(tournamentIncludedModels)).length < 2}>{tournamentRunning ? 'Running...' : 'Run Full Tournament'}</button><button onClick={stopTournament} disabled={!tournamentRunning}>Stop</button></div>
            {tournamentResult && (
              <p className="muted">
                Tournament complete: {tournamentResult.games} games | Agents: {matrixLabels.length}
              </p>
            )}
            <div className="logBox">
              {tournamentLogs.length === 0 && <div className="muted">No game logs yet.</div>}
              {tournamentLogs.map((g, idx) => (
                <div key={`${g.game}-${idx}`} className="logLine">
                  Game {g.game}/{g.games}: {g.x_model} (X) vs {g.o_model} (O) | starter {g.starter} | moves {g.moves} | winner {g.winner}
                </div>
              ))}
            </div>
          </>}
        </aside>

        <section className="contentArea">
          {tab === 'play' && <div className="grid">
            <div className="card"><h3>Interactive Board</h3><p className={gameResult ? 'resultBanner done' : 'resultBanner'}>{gameMode === 'user_vs_ai' && !gameStarted ? `Ready: click Start Game (${userSide} as user)` : playStatusText}</p><div className="board">{symbols.map((s, i) => <button key={i} className="cell" onClick={() => playHuman(i)} disabled={board[i] !== 0 || Boolean(gameResult) || gameMode !== 'user_vs_ai' || !gameStarted || turn !== userMarker}>{s}</button>)}</div></div>
            <div className="card"><h3>AI Brain Visualizer</h3>{isAiTurn ? null : <><p className="muted">Previous AI move probabilities (last AI turn).</p>{brain.length === 0 ? <div className="muted">No AI move yet.</div> : brain.map((m, i) => <div key={i}>#{i + 1} cell <b>{m.action + 1}</b> (index {m.action}) | prob {(m.probability * 100).toFixed(2)}% | q {m.q_value.toFixed(4)}</div>)}</>}</div>
          </div>}

          {tab === 'training' && <div className="grid">
            <div className="card full trainingMetricsCard">
              <h3>Live Metrics</h3>
              <div className="metricsChartWrap">
                <canvas ref={chartRef} height="420"></canvas>
              </div>
              <p className="muted metricsFooter">Last metric: {lastMetric ? `ep ${lastMetric.episode}, train ${Number(lastMetric.loss).toFixed(4)}, val ${Number(lastMetric.val_loss ?? lastMetric.loss).toFixed(4)}` : 'none'}</p>
            </div>
          </div>}

          {tab === 'tournament' && <div className="grid">
            <div className="card full mainPanel">
              <h3>Rankings (Random Matchmaking)</h3>
              <div className="tableWrap">
                <table>
                  <thead><tr><th>#</th><th>Model</th><th>W</th><th>L</th><th>D</th><th>Games</th><th>Score</th><th>Win Rate</th></tr></thead>
                  <tbody>
                    {leaderboard.map((r, i) => <tr key={r.model_id}><td><span className={`rankBadge r${Math.min(i + 1, 3)}`}>{i + 1}</span></td><td>{r.model_id}</td><td>{r.wins ?? ((r.wins_as_x || 0) + (r.wins_as_o || 0))}</td><td>{r.losses ?? 0}</td><td>{r.draws}</td><td>{r.total_games}</td><td>{(r.score * 100).toFixed(1)}%</td><td>{(r.win_rate * 100).toFixed(1)}%</td></tr>)}
                    {leaderboard.length === 0 && <tr><td colSpan="8">No ranking yet.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card full mainPanel">
              <h3>Head-to-Head Matrix</h3>
              <p className="muted">Axis: <b>X ↓</b> (row) vs <b>O →</b> (column). Cell format: <b>W/D/L</b> from row-X perspective.</p>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>X ↓ \ O →</th>
                      {matrixLabels.map((lbl) => <th key={`h-${lbl}`}>{lbl}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {matrixLabels.map((rowLbl, i) => (
                      <tr key={`r-${rowLbl}`}>
                        <td>{rowLbl}</td>
                        {matrixLabels.map((colLbl, j) => {
                          const c = matrixData?.[i]?.[j]
                          if (!c || c.games === 0) return <td key={`c-${i}-${j}`}>-</td>
                          return <td key={`c-${i}-${j}`}>{c.wins}/{c.draws}/{c.losses}</td>
                        })}
                      </tr>
                    ))}
                    {matrixLabels.length === 0 && <tr><td colSpan="2">No matrix yet.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>}
        </section>
      </div>
      {toast && <div className="snackbar">{toast}</div>}
    </div>
  )
}

export default App

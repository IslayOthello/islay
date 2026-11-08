// Islay Othello — Professional Desktop Client & Board Renderer

let state = {
  grid: Array(8).fill(null).map(() => Array(8).fill(0)),
  disc_move_numbers: Array(8).fill(null).map(() => Array(8).fill(0)),
  turn: 1, // 1: Black, 2: White
  rule: "Othello",
  mode: 1, // 1: vs AI (B), 2: vs AI (W), 3: 2-Player, 4: Self-Play
  b_count: 2,
  w_count: 2,
  empty_count: 60,
  game_over: false,
  winner: null,
  forced_pass: false,
  current_ply: 0,
  history: [],
  last_move: null,
  legal_moves: {},
  opening_name: "Standard Start Position",
  setup_mode: false,
  hint_move: null,
  is_thinking: false,
  engine_stats: { nodes: 0, nps: 0, time: 0, pv: "", value: 0, evaluations: 0, state: "ready" },
  status_msg: "Engine: islay 0.1.0 ready"
};

// UI Settings
let showNumbers = true;
let showEvals = true;
let showGraph = true;
let setupTool = 1;
let activeTab = "moves";
let hoveredSquare = null;

// Precision Board Geometry Constants
const CANVAS_SIZE = 560;
const MARGIN = 24;
const GRID_SIZE = 512;
const CS = 64; // 64 * 8 = 512px

const boardCanvas = document.getElementById("board-canvas");
const bCtx = boardCanvas.getContext("2d");

const graphCanvas = document.getElementById("graph-canvas");
const gCtx = graphCanvas.getContext("2d");

function initDpi(canvas, ctx, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}

initDpi(boardCanvas, bCtx, CANVAS_SIZE, CANVAS_SIZE);
initDpi(graphCanvas, gCtx, 436, 80);

// --- IPC Bridge to Lua Backend ---
function sendCmd(cmd, payload = {}) {
  const msg = { cmd, ...payload };
  if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.islay) {
    window.webkit.messageHandlers.islay.postMessage(JSON.stringify(msg));
  } else {
    fetch("/api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(msg)
    })
    .then(r => r.json())
    .then(data => onStateUpdate(data))
    .catch(() => {});
  }
}

window.onIslayMessage = function(raw) {
  const data = (typeof raw === "string") ? JSON.parse(raw) : raw;
  onStateUpdate(data);
};

function onStateUpdate(data) {
  state = data;
  renderAll();
}

// --- Board Canvas Graphics Rendering ---
function renderBoard() {
  // 1. Satin Dark Outer Bezel
  bCtx.fillStyle = "#141816";
  bCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Bezel inner groove
  bCtx.strokeStyle = "#090d0a";
  bCtx.lineWidth = 1;
  bCtx.strokeRect(MARGIN - 1.5, MARGIN - 1.5, GRID_SIZE + 3, GRID_SIZE + 3);

  // 2. Coordinate Labels (Centered in outer margins, outside playing felt)
  bCtx.fillStyle = "#839487";
  bCtx.font = "600 11px ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace";
  bCtx.textAlign = "center";
  bCtx.textBaseline = "middle";

  for (let i = 0; i < 8; i++) {
    const colName = String.fromCharCode(65 + i);
    const rowName = String(i + 1);
    const centerOffset = MARGIN + (i + 0.5) * CS;

    // Top & Bottom Columns A-H
    bCtx.fillText(colName, centerOffset, 13);
    bCtx.fillText(colName, centerOffset, CANVAS_SIZE - 11);

    // Left & Right Ranks 1-8
    bCtx.fillText(rowName, 12, centerOffset);
    bCtx.fillText(rowName, CANVAS_SIZE - 12, centerOffset);
  }

  // 3. Tournament Emerald Felt
  bCtx.fillStyle = "#1b6838";
  bCtx.fillRect(MARGIN, MARGIN, GRID_SIZE, GRID_SIZE);

  // Subtle checkered grid texture
  bCtx.fillStyle = "rgba(0, 0, 0, 0.025)";
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      if ((x + y) % 2 === 1) {
        bCtx.fillRect(MARGIN + x * CS, MARGIN + y * CS, CS, CS);
      }
    }
  }

  // 4. Precision Grid Lines
  bCtx.strokeStyle = "#0e4120";
  bCtx.lineWidth = 1.2;
  for (let i = 0; i <= 8; i++) {
    const pos = MARGIN + i * CS;
    // Vertical
    bCtx.beginPath();
    bCtx.moveTo(pos, MARGIN);
    bCtx.lineTo(pos, MARGIN + GRID_SIZE);
    bCtx.stroke();

    // Horizontal
    bCtx.beginPath();
    bCtx.moveTo(MARGIN, pos);
    bCtx.lineTo(MARGIN + GRID_SIZE, pos);
    bCtx.stroke();
  }

  // 5. Tournament Star Points at C3, F3, C6, F6 (grid lines 2 and 6)
  const starGridIntersections = [
    [2, 2], [6, 2], [2, 6], [6, 6]
  ];
  starGridIntersections.forEach(([sx, sy]) => {
    const px = MARGIN + sx * CS;
    const py = MARGIN + sy * CS;

    // Base rivet
    bCtx.fillStyle = "#082613";
    bCtx.beginPath();
    bCtx.arc(px, py, 3.5, 0, Math.PI * 2);
    bCtx.fill();

    // Center highlight
    bCtx.fillStyle = "#27ae60";
    bCtx.beginPath();
    bCtx.arc(px, py, 1.2, 0, Math.PI * 2);
    bCtx.fill();
  });

  // Calculate candidate flip previews on hover
  const previewFlips = new Set();
  if (hoveredSquare && !state.is_thinking && !state.setup_mode) {
    const uci = coordToUci(hoveredSquare.x, hoveredSquare.y);
    const m = state.legal_moves[uci];
    if (m && m.flips) {
      m.flips.forEach(f => previewFlips.add(`${f.x}_${f.y}`));
    }
  }

  // 6. Draw Discs
  for (let y = 1; y <= 8; y++) {
    for (let x = 1; x <= 8; x++) {
      const val = state.grid[y - 1][x - 1];
      const moveNum = state.disc_move_numbers[y - 1][x - 1];
      const cx = MARGIN + (x - 0.5) * CS;
      const cy = MARGIN + (y - 0.5) * CS;
      const r = CS * 0.415; // 26.56px

      if (val === 1 || val === 2) {
        drawDisc(cx, cy, r, val, moveNum);
        if (previewFlips.has(`${x}_${y}`)) {
          bCtx.strokeStyle = "rgba(245, 158, 11, 0.95)";
          bCtx.lineWidth = 2.5;
          bCtx.beginPath();
          bCtx.arc(cx, cy, r * 0.58, 0, Math.PI * 2);
          bCtx.stroke();
        }
      }
    }
  }

  // 7. Last Move Highlight (Amber Ring)
  if (state.last_move && state.last_move.x > 0) {
    const lx = MARGIN + (state.last_move.x - 0.5) * CS;
    const ly = MARGIN + (state.last_move.y - 0.5) * CS;
    bCtx.strokeStyle = "#f59e0b";
    bCtx.lineWidth = 2.5;
    bCtx.beginPath();
    bCtx.arc(lx, ly, CS * 0.44, 0, Math.PI * 2);
    bCtx.stroke();
  }

  // 8. Engine Suggested Move (Cyan Ring)
  if (state.hint_move && state.hint_move !== "pass" && state.hint_move !== "0000") {
    const hx = state.hint_move.charCodeAt(0) - 96;
    const hy = parseInt(state.hint_move.charAt(1));
    const hcx = MARGIN + (hx - 0.5) * CS;
    const hcy = MARGIN + (hy - 0.5) * CS;
    bCtx.strokeStyle = "#06b6d4";
    bCtx.lineWidth = 2.5;
    bCtx.beginPath();
    bCtx.arc(hcx, hcy, CS * 0.44, 0, Math.PI * 2);
    bCtx.stroke();
  }

  // 9. Legal Move Indicators & Candidate Evaluations
  if (!state.game_over && !state.setup_mode) {
    for (const [uci, m] of Object.entries(state.legal_moves || {})) {
      const cx = MARGIN + (m.x - 0.5) * CS;
      const cy = MARGIN + (m.y - 0.5) * CS;
      const isHovered = hoveredSquare && hoveredSquare.x === m.x && hoveredSquare.y === m.y;

      if (isHovered) {
        bCtx.fillStyle = "rgba(255, 255, 255, 0.32)";
        bCtx.beginPath();
        bCtx.arc(cx, cy, CS * 0.38, 0, Math.PI * 2);
        bCtx.fill();

        bCtx.fillStyle = "#ffffff";
        bCtx.font = "bold 13px -apple-system, BlinkMacSystemFont, sans-serif";
        bCtx.textAlign = "center";
        bCtx.textBaseline = "middle";
        bCtx.fillText(`+${m.flips ? m.flips.length : 0}`, cx, cy);
      } else {
        bCtx.fillStyle = "rgba(255, 255, 255, 0.45)";
        bCtx.beginPath();
        bCtx.arc(cx, cy, 5, 0, Math.PI * 2);
        bCtx.fill();
      }

      // WZebra Numerical Evaluation Badge
      if (showEvals && m.eval !== undefined && m.eval !== null) {
        const score = m.eval;
        const scoreStr = (score >= 0 ? "+" : "") + score.toFixed(1);
        bCtx.fillStyle = score >= 0 ? "rgba(22, 163, 74, 0.92)" : "rgba(220, 38, 38, 0.92)";
        bCtx.beginPath();
        bCtx.roundRect(cx - 15, cy + 9, 30, 13, 3);
        bCtx.fill();

        bCtx.fillStyle = "#ffffff";
        bCtx.font = "bold 9px ui-monospace, 'SF Mono', monospace";
        bCtx.textAlign = "center";
        bCtx.textBaseline = "middle";
        bCtx.fillText(scoreStr, cx, cy + 15.5);
      }
    }
  }
}

// 3D Realistic Disc Rendering
function drawDisc(cx, cy, r, colorId, moveNum) {
  const isBlack = (colorId === 1);

  // 1. Soft contact shadow
  bCtx.fillStyle = "rgba(0, 0, 0, 0.32)";
  bCtx.beginPath();
  bCtx.arc(cx, cy + 2.5, r, 0, Math.PI * 2);
  bCtx.fill();

  // 2. Outer rim
  bCtx.fillStyle = isBlack ? "#000000" : "#cbd5e1";
  bCtx.beginPath();
  bCtx.arc(cx, cy, r, 0, Math.PI * 2);
  bCtx.fill();

  // 3. Tactile body gradient
  const grad = bCtx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r);
  if (isBlack) {
    grad.addColorStop(0, "#4b5563");
    grad.addColorStop(0.5, "#1f2937");
    grad.addColorStop(1, "#0b1120");
  } else {
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(0.65, "#f8fafc");
    grad.addColorStop(1, "#e2e8f0");
  }

  bCtx.fillStyle = grad;
  bCtx.beginPath();
  bCtx.arc(cx, cy, r - 1.2, 0, Math.PI * 2);
  bCtx.fill();

  // 4. Centered move numbers (WZebra feature)
  if (showNumbers && moveNum && moveNum > 0) {
    bCtx.fillStyle = isBlack ? "#ffffff" : "#0f172a";
    bCtx.font = "bold 13px ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace";
    bCtx.textAlign = "center";
    bCtx.textBaseline = "middle";
    bCtx.fillText(String(moveNum), cx, cy);
  }
}

// --- WZebra Advantage Curve Graph ---
function renderGraph() {
  const wrap = graphCanvas.parentElement;
  const w = wrap.clientWidth || 436;
  const h = wrap.clientHeight || 80;

  const dpr = window.devicePixelRatio || 1;
  if (graphCanvas.width !== w * dpr || graphCanvas.height !== h * dpr) {
    initDpi(graphCanvas, gCtx, w, h);
  }

  gCtx.fillStyle = "#ffffff";
  gCtx.fillRect(0, 0, w, h);

  const zeroY = h / 2;

  // Grid lines
  gCtx.strokeStyle = "#e5e7eb";
  gCtx.lineWidth = 1;
  gCtx.beginPath();
  gCtx.moveTo(0, zeroY);
  gCtx.lineTo(w, zeroY);
  gCtx.stroke();

  // Scale labels
  gCtx.fillStyle = "#9ca3af";
  gCtx.font = "9px ui-monospace, 'SF Mono', monospace";
  gCtx.textAlign = "left";
  gCtx.fillText("+1", 5, 11);
  gCtx.fillText(" 0", 5, zeroY + 3);
  gCtx.fillText("-1", 5, h - 4);

  const history = state.history || [];
  if (history.length === 0) return;

  const maxPlies = Math.max(60, history.length);
  const stepX = (w - 36) / maxPlies;
  const offsetX = 28;

  const points = [];
  history.forEach((rec, idx) => {
    const px = offsetX + idx * stepX;
    const score = Math.max(-1.0, Math.min(1.0, rec.eval_score || 0.0));
    const py = zeroY - score * (zeroY - 8);
    points.push({ x: px, y: py, ply: rec.ply });
  });

  // Soft gradient under curve
  if (points.length >= 2) {
    const gradArea = gCtx.createLinearGradient(0, 0, 0, h);
    gradArea.addColorStop(0, "rgba(9, 105, 218, 0.12)");
    gradArea.addColorStop(0.5, "rgba(9, 105, 218, 0.01)");
    gradArea.addColorStop(1, "rgba(207, 34, 46, 0.12)");

    gCtx.fillStyle = gradArea;
    gCtx.beginPath();
    gCtx.moveTo(points[0].x, zeroY);
    points.forEach(pt => gCtx.lineTo(pt.x, pt.y));
    gCtx.lineTo(points[points.length - 1].x, zeroY);
    gCtx.closePath();
    gCtx.fill();

    // Curve Line
    gCtx.strokeStyle = "#0969da";
    gCtx.lineWidth = 2;
    gCtx.beginPath();
    gCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      gCtx.lineTo(points[i].x, points[i].y);
    }
    gCtx.stroke();
  }

  // Data Points
  points.forEach(pt => {
    const isCur = (pt.ply === state.current_ply);
    gCtx.fillStyle = isCur ? "#d97706" : "#0969da";
    gCtx.beginPath();
    gCtx.arc(pt.x, pt.y, isCur ? 4 : 2, 0, Math.PI * 2);
    gCtx.fill();
  });
}

// Click graph to jump to move
graphCanvas.addEventListener("click", e => {
  const rect = graphCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const history = state.history || [];
  if (history.length === 0) return;

  const w = rect.width;
  const maxPlies = Math.max(60, history.length);
  const stepX = (w - 36) / maxPlies;
  const offsetX = 28;

  let closestPly = 0;
  let minDist = 9999;
  history.forEach((rec, idx) => {
    const px = offsetX + idx * stepX;
    const dist = Math.abs(mx - px);
    if (dist < minDist) {
      minDist = dist;
      closestPly = rec.ply;
    }
  });

  if (closestPly >= 0) {
    sendCmd("goto_ply", { ply: closestPly });
  }
});

// --- Move Notation Table ---
function renderMoveTable() {
  const tbody = document.getElementById("move-tbody");
  tbody.innerHTML = "";

  const history = state.history || [];
  const totalMoves = Math.ceil(history.length / 2);

  for (let m = 1; m <= totalMoves; m++) {
    const bIdx = (m - 1) * 2;
    const wIdx = bIdx + 1;
    const bRec = history[bIdx];
    const wRec = history[wIdx];

    const tr = document.createElement("tr");

    const tdNum = document.createElement("td");
    tdNum.className = "col-num";
    tdNum.textContent = `${m}.`;
    tr.appendChild(tdNum);

    const tdB = document.createElement("td");
    tdB.className = "col-ply";
    if (bRec) {
      tdB.textContent = bRec.uci.toUpperCase();
      if (state.current_ply === bRec.ply) tdB.classList.add("active");
      tdB.onclick = () => sendCmd("goto_ply", { ply: bRec.ply });
    }
    tr.appendChild(tdB);

    const tdW = document.createElement("td");
    tdW.className = "col-ply";
    if (wRec) {
      tdW.textContent = wRec.uci.toUpperCase();
      if (state.current_ply === wRec.ply) tdW.classList.add("active");
      tdW.onclick = () => sendCmd("goto_ply", { ply: wRec.ply });
    }
    tr.appendChild(tdW);

    const tdEval = document.createElement("td");
    tdEval.className = "col-eval";
    const rec = wRec || bRec;
    if (rec && rec.eval_score !== undefined) {
      const score = rec.eval_score;
      tdEval.textContent = (score >= 0 ? "+" : "") + score.toFixed(2);
      tdEval.style.color = (score >= 0) ? "var(--color-success)" : "var(--color-danger)";
      tdEval.style.fontWeight = "600";
    }
    tr.appendChild(tdEval);

    tbody.appendChild(tr);
  }
}

// --- Candidate Moves Ranking List ---
function renderCandidates() {
  const list = document.getElementById("candidates-list");
  list.innerHTML = "";

  const candidates = [];
  for (const [uci, m] of Object.entries(state.legal_moves || {})) {
    candidates.push({ uci: uci.toUpperCase(), eval: m.eval || 0, flips: m.flips ? m.flips.length : 0 });
  }

  candidates.sort((a, b) => b.eval - a.eval);

  if (candidates.length === 0) {
    list.innerHTML = `<div class="candidate-empty">No legal candidate moves</div>`;
    return;
  }

  candidates.forEach((c, idx) => {
    const row = document.createElement("div");
    row.className = "candidate-row";
    row.innerHTML = `
      <span><b>#${idx + 1}</b> ${c.uci} (+${c.flips} discs)</span>
      <span style="color: ${c.eval >= 0 ? 'var(--color-success)' : 'var(--color-danger)'}; font-weight: 600;">
        ${c.eval >= 0 ? '+' : ''}${c.eval.toFixed(2)}
      </span>
    `;
    list.appendChild(row);
  });
}

// --- Render Full State ---
function renderAll() {
  renderBoard();
  renderGraph();
  renderMoveTable();
  renderCandidates();

  // Header info
  document.getElementById("opening-name").textContent = state.opening_name || "Standard Opening";
  document.getElementById("ply-indicator").textContent = `Move ${state.current_ply}`;

  // Match score box
  document.getElementById("score-black").textContent = state.b_count;
  document.getElementById("score-white").textContent = state.w_count;

  const isBTurn = (state.turn === 1 && !state.game_over);
  const isWTurn = (state.turn === 2 && !state.game_over);

  const cardB = document.getElementById("player-card-black");
  const cardW = document.getElementById("player-card-white");
  if (cardB) cardB.classList.toggle("active", isBTurn);
  if (cardW) cardW.classList.toggle("active", isWTurn);

  // Subtitles based on game mode
  const typeB = document.getElementById("type-black");
  const typeW = document.getElementById("type-white");
  if (state.mode === 1) {
    if (typeB) typeB.textContent = "Human";
    if (typeW) typeW.textContent = "islay 0.1.0";
  } else if (state.mode === 2) {
    if (typeB) typeB.textContent = "islay 0.1.0";
    if (typeW) typeW.textContent = "Human";
  } else if (state.mode === 3) {
    if (typeB) typeB.textContent = "Player 1";
    if (typeW) typeW.textContent = "Player 2";
  } else if (state.mode === 4) {
    if (typeB) typeB.textContent = "islay (Black)";
    if (typeW) typeW.textContent = "islay (White)";
  }

  // Advantage Vertical Gauge
  const val = Math.max(-1.0, Math.min(1.0, state.engine_stats.value || 0.0));
  const blackRatio = (val + 1.0) / 2.0; // 0.0 (all white) to 1.0 (all black)
  const blackPct = Math.max(2, Math.min(98, blackRatio * 100));
  const whitePct = 100 - blackPct;

  const gB = document.getElementById("gauge-black");
  const gW = document.getElementById("gauge-white");
  if (gB) gB.style.height = `${blackPct.toFixed(1)}%`;
  if (gW) gW.style.height = `${whitePct.toFixed(1)}%`;
  const gScore = document.getElementById("gauge-score");
  if (gScore) gScore.textContent = (val >= 0 ? "+" : "") + val.toFixed(1);

  // Engine Stats
  const s = state.engine_stats || {};
  document.getElementById("stat-nodes").textContent = (s.nodes || 0).toLocaleString();
  document.getElementById("stat-nps").textContent = (s.nps || 0).toLocaleString() + " NPS";
  document.getElementById("stat-time").textContent = (s.time || 0) + " ms";
  document.getElementById("stat-pv").textContent = s.pv ? s.pv.toUpperCase() : "(none)";
  const statEval = document.getElementById("stat-eval");
  if (statEval) {
    const v = s.value || 0;
    statEval.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
    statEval.style.color = (v >= 0) ? "var(--color-success)" : "var(--color-danger)";
  }

  // Status Bar
  document.getElementById("status-banner").textContent = state.status_msg || "Ready";
  document.getElementById("status-engine-pill").textContent = `● ${s.state ? s.state.toUpperCase() : "READY"}`;

  // If in setup mode, auto-switch to setup tab
  if (state.setup_mode && activeTab !== "setup") {
    switchTab("setup");
  }
}

// --- Mouse Interaction ---
boardCanvas.addEventListener("mousemove", e => {
  const rect = boardCanvas.getBoundingClientRect();
  const scaleX = CANVAS_SIZE / rect.width;
  const scaleY = CANVAS_SIZE / rect.height;
  const mx = (e.clientX - rect.left) * scaleX - MARGIN;
  const my = (e.clientY - rect.top) * scaleY - MARGIN;

  if (mx >= 0 && mx < GRID_SIZE && my >= 0 && my < GRID_SIZE) {
    const cx = Math.floor(mx / CS) + 1;
    const cy = Math.floor(my / CS) + 1;
    hoveredSquare = { x: cx, y: cy };
    document.getElementById("status-hover").textContent = `Square: ${String.fromCharCode(64 + cx)}${cy}`;
  } else {
    hoveredSquare = null;
    document.getElementById("status-hover").textContent = "Square: —";
  }
  renderBoard();
});

boardCanvas.addEventListener("mouseleave", () => {
  hoveredSquare = null;
  document.getElementById("status-hover").textContent = "Square: —";
  renderBoard();
});

boardCanvas.addEventListener("click", () => {
  if (!hoveredSquare) return;

  if (state.setup_mode) {
    sendCmd("setup_set", { x: hoveredSquare.x, y: hoveredSquare.y, color: setupTool });
  } else {
    sendCmd("move", { x: hoveredSquare.x, y: hoveredSquare.y });
  }
});

function coordToUci(x, y) {
  return String.fromCharCode(96 + x) + String(y);
}

// Tab Switching
function switchTab(tabId) {
  activeTab = tabId;
  document.querySelectorAll(".tab-chip").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(c => c.style.display = "none");

  const btn = document.getElementById(`tab-btn-${tabId}`);
  const pane = document.getElementById(`pane-${tabId}`);
  if (btn) btn.classList.add("active");
  if (pane) pane.style.display = "flex";

  if (tabId === "setup" && !state.setup_mode) {
    sendCmd("setup_toggle");
  }
}

function finishSetup() {
  sendCmd("setup_toggle");
  switchTab("moves");
}

// Toolbar Dropdown Menu
function toggleQuickMenu(event) {
  if (event) event.stopPropagation();
  const pane = document.getElementById("quick-menu-pane");
  if (pane) {
    pane.classList.toggle("open");
  }
}

window.addEventListener("click", e => {
  if (!e.target.closest(".tool-dropdown")) {
    document.querySelectorAll(".dropdown-pane").forEach(el => el.classList.remove("open"));
  }
});

// Prevent default browser right-click context menu and trigger Hint
boardCanvas.addEventListener("contextmenu", e => {
  e.preventDefault();
  sendCmd("hint");
});

// UI Actions
function toggleNumbers() {
  showNumbers = !showNumbers;
  const btn = document.getElementById("btn-toggle-num");
  if (btn) btn.classList.toggle("active", showNumbers);
  const chk = document.getElementById("chk-num");
  if (chk) chk.style.display = showNumbers ? "inline" : "none";
  renderBoard();
}

function toggleEvals() {
  showEvals = !showEvals;
  const btn = document.getElementById("btn-toggle-eval");
  if (btn) btn.classList.toggle("active", showEvals);
  const chk = document.getElementById("chk-eval");
  if (chk) chk.style.display = showEvals ? "inline" : "none";
  renderBoard();
}

function toggleGraph() {
  showGraph = !showGraph;
  const btn = document.getElementById("btn-toggle-graph");
  if (btn) btn.classList.toggle("active", showGraph);
  const pnl = document.getElementById("curve-panel");
  if (pnl) pnl.style.display = showGraph ? "block" : "none";
  const chk = document.getElementById("chk-graph");
  if (chk) chk.style.display = showGraph ? "inline" : "none";
}

function setMode(m) {
  sendCmd("set_mode", { mode: m });
}

function setRule(r) {
  sendCmd("set_rule", { rule: r });
}

function setBudget(idx) {
  sendCmd("set_budget", { idx });
}

function gotoPly(p) {
  sendCmd("goto_ply", { ply: p });
}

function gotoEnd() {
  sendCmd("goto_ply", { ply: (state.history || []).length });
}

function setSetupTool(t) {
  setupTool = t;
  document.querySelectorAll(".setup-chip-row .tool-chip").forEach(b => b.classList.remove("active"));
  if (t === 1) document.getElementById("chip-black").classList.add("active");
  if (t === 2) document.getElementById("chip-white").classList.add("active");
  if (t === 0) document.getElementById("chip-empty").classList.add("active");
}

function toggleSetupTurn() {
  const nextTurn = (state.turn === 1) ? 2 : 1;
  document.getElementById("btn-setup-turn").textContent = (nextTurn === 1) ? "Turn: Black" : "Turn: White";
  sendCmd("setup_turn", { turn: nextTurn });
}

function copyFen() {
  if (navigator.clipboard) {
    navigator.clipboard.writeText(state.fen);
    alert("Position FEN copied:\n" + state.fen);
  }
}

function pasteFen() {
  const fen = prompt("Paste 64-character diagram and side to move:");
  if (fen && fen.length >= 64) {
    sendCmd("load_fen", { fen: fen.slice(0, 64), stm: fen.slice(64).trim() || "X" });
  }
}

function triggerPerft() {
  const d = prompt("Enter Perft Depth (1-8):", "5");
  if (d) {
    sendCmd("perft", { depth: parseInt(d) });
  }
}

// Window resize listener to keep curve crisp
window.addEventListener("resize", () => {
  renderGraph();
});

// Initial state query
sendCmd("init");
setInterval(() => {
  if (state.is_thinking) {
    sendCmd("get_state");
  }
}, 300);

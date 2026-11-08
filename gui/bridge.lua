-- Lua Backend Bridge for Islay Desktop GUI
local board    = require("gui.board")
local bitboard = require("gui.bitboard")
local engine   = require("gui.engine")
local openings = require("gui.openings")
local json     = require("gui.json")

local state = {
  b = board.new("Othello"),
  eng = engine.new(),
  mode = 1, -- 1: vs AI (B), 2: vs AI (W), 3: 2-Player, 4: Self-Play
  budgets = {
    { label = "1K nodes",  nodes = 1000 },
    { label = "10K nodes", nodes = 10000 },
    { label = "50K nodes", nodes = 50000 },
    { label = "500 ms",    movetime = 500 }
  },
  selected_budget = 2,
  hint_move = nil,
  is_thinking = false,
  status_msg = "Ready"
}

state.eng.on_ready = function()
  state.status_msg = "Engine ready: " .. (state.eng.binary_path or "islay")
end

state.eng.on_error = function(err)
  state.status_msg = "Engine error: " .. tostring(err)
end

state.eng:start()

local function get_full_state()
  local b = state.b
  local b_count, w_count, empty_count = b:get_counts()
  local legal_moves = {}
  if not b.game_over and not b.setup_mode then
    for uci, m in pairs(b:get_legal_moves(b.turn)) do
      legal_moves[uci] = {
        x = m.x,
        y = m.y,
        flips = m.flips,
        eval = b.legal_evals[uci] or nil
      }
    end
  end

  local history_list = {}
  for i = 1, #b.history do
    local h = b.history[i]
    table.insert(history_list, {
      ply = h.ply,
      uci = h.uci,
      color = h.color,
      eval_score = h.eval_score or 0.0
    })
  end

  local fen, stm = b:to_fen()

  return {
    grid = b.grid,
    disc_move_numbers = b.disc_move_numbers,
    turn = b.turn,
    rule = b.rule,
    mode = state.mode,
    b_count = b_count,
    w_count = w_count,
    empty_count = empty_count,
    game_over = b.game_over,
    winner = b.winner,
    forced_pass = b.forced_pass,
    current_ply = b.current_ply,
    history = history_list,
    last_move = b.last_move,
    legal_moves = legal_moves,
    opening_name = b:get_opening_name(),
    setup_mode = b.setup_mode,
    hint_move = state.hint_move,
    is_thinking = state.is_thinking,
    fen = fen .. " " .. stm,
    engine_stats = {
      nodes = state.eng.stats.nodes,
      nps = state.eng.stats.nps,
      time = state.eng.stats.time,
      pv = state.eng.stats.pv,
      value = state.eng.stats.value,
      evaluations = state.eng.stats.evaluations,
      state = state.eng.state
    },
    status_msg = state.status_msg
  }
end

local function handle_cmd(req)
  local cmd = req.cmd
  local b = state.b

  if cmd == "init" or cmd == "get_state" then
    return get_full_state()

  elseif cmd == "new_game" then
    b:reset()
    state.hint_move = nil
    state.eng:sync_position("")
    state.status_msg = "New game started"
    if (state.mode == 2 or state.mode == 4) and not b.game_over then
      handle_cmd({ cmd = "trigger_ai" })
    end
    return get_full_state()

  elseif cmd == "move" then
    local x, y = req.x, req.y
    local ok, err = b:apply_move(x, y, state.eng.stats.value or 0.0)
    if ok then
      state.hint_move = nil
      state.eng:sync_position(b:to_uci_moves())
      state.status_msg = "Played " .. board.to_uci(x, y):upper()
      if (state.mode == 1 and b.turn == board.WHITE) or
         (state.mode == 2 and b.turn == board.BLACK) or
         (state.mode == 4) then
        if not b.game_over then
          handle_cmd({ cmd = "trigger_ai" })
        end
      end
    else
      state.status_msg = "Invalid move: " .. tostring(err)
    end
    return get_full_state()

  elseif cmd == "trigger_ai" then
    if b.game_over or state.is_thinking then return get_full_state() end
    state.is_thinking = true
    state.status_msg = "Islay is thinking..."
    state.eng:sync_position(b:to_uci_moves())

    local budget = state.budgets[state.selected_budget]
    state.eng.on_bestmove = function(bm, stats)
      state.is_thinking = false
      state.eng.on_bestmove = nil

      if bm and bm ~= "0000" then
        -- Rigorous verification of engine move correctness using ported bitboard model
        local is_legal, verify_err = b:verify_move(bm)
        if not is_legal then
          state.status_msg = string.format("[ENGINE ERROR] Illegal move '%s': %s", tostring(bm), tostring(verify_err))
          io.stderr:write(string.format("[CRITICAL] Engine returned illegal move '%s': %s (Position: %s)\n",
            tostring(bm), tostring(verify_err), b:to_uci_moves()))
        else
          if bm == "pass" then
            local ok, err = b:pass(stats.value or 0.0)
            if ok then
              state.status_msg = "Islay passed"
            else
              state.status_msg = "Pass failed: " .. tostring(err)
            end
          else
            local x, y = board.from_uci(bm)
            if x and y then
              local ok, err = b:apply_move(x, y, stats.value or 0.0)
              if ok then
                state.status_msg = "Islay played " .. bm:upper()
              else
                state.status_msg = "Move application failed: " .. tostring(err)
              end
            end
          end
        end
      end

      -- If Self-play and game continues, trigger next AI move
      if state.mode == 4 and not b.game_over then
        handle_cmd({ cmd = "trigger_ai" })
      end
    end

    if budget.movetime then
      state.eng:search_movetime(budget.movetime)
    else
      state.eng:search_nodes(budget.nodes)
    end
    return get_full_state()

  elseif cmd == "pass" then
    local ok, err = b:pass()
    if ok then
      state.eng:sync_position(b:to_uci_moves())
      state.status_msg = "Passed turn"
    else
      state.status_msg = "Cannot pass: " .. tostring(err)
    end
    return get_full_state()

  elseif cmd == "undo" then
    if state.mode == 1 or state.mode == 2 then
      b:undo()
      b:undo()
    else
      b:undo()
    end
    state.hint_move = nil
    state.eng:sync_position(b:to_uci_moves())
    state.status_msg = "Undid move"
    return get_full_state()

  elseif cmd == "redo" then
    b:redo()
    state.hint_move = nil
    state.eng:sync_position(b:to_uci_moves())
    return get_full_state()

  elseif cmd == "goto_ply" then
    b:goto_ply(req.ply or 0)
    state.hint_move = nil
    state.eng:sync_position(b:to_uci_moves())
    return get_full_state()

  elseif cmd == "hint" then
    state.eng:sync_position(b:to_uci_moves())
    state.eng.on_bestmove = function(bm, stats)
      state.eng.on_bestmove = nil
      if bm and bm ~= "0000" then
        -- Verify engine suggested hint correctness
        local is_legal, verify_err = b:verify_move(bm)
        if is_legal then
          state.hint_move = bm
          state.status_msg = "Suggested: " .. tostring(bm):upper()
        else
          state.hint_move = nil
          state.status_msg = string.format("Engine suggested illegal move '%s': %s", tostring(bm), tostring(verify_err))
        end
      end
    end
    state.eng:search_nodes(5000)
    return get_full_state()

  elseif cmd == "perft" then
    local d = req.depth or 5
    state.status_msg = "Running Perft depth " .. d .. "..."
    state.eng:sync_position(b:to_uci_moves())
    state.eng.on_perft_done = function(res)
      state.eng.on_perft_done = nil
      local lua_nodes = bitboard.perft(b.rep, d, b.rule)
      if lua_nodes == res.nodes then
        state.status_msg = string.format("Perft %d: %s nodes (Verified 100%%) in %s (%s)",
          res.depth, tostring(res.nodes), res.time_str, res.speed_str)
      else
        state.status_msg = string.format("Perft %d MISMATCH: Engine=%s vs Lua=%d",
          res.depth, tostring(res.nodes), lua_nodes)
      end
    end
    state.eng:perft(d)
    return get_full_state()

  elseif cmd == "set_mode" then
    state.mode = req.mode or 1
    state.status_msg = "Game mode updated"
    if (state.mode == 2 or state.mode == 4) and not b.game_over then
      handle_cmd({ cmd = "trigger_ai" })
    end
    return get_full_state()

  elseif cmd == "set_budget" then
    state.selected_budget = req.idx or 2
    state.status_msg = "AI Level: " .. state.budgets[state.selected_budget].label
    return get_full_state()

  elseif cmd == "set_rule" then
    local r = req.rule or "Othello"
    b.rule = r
    state.eng:set_rule(r)
    state.status_msg = "Rule set to " .. r
    return get_full_state()

  elseif cmd == "setup_toggle" then
    b.setup_mode = not b.setup_mode
    if not b.setup_mode then
      state.eng:sync_position(b:to_uci_moves())
    end
    return get_full_state()

  elseif cmd == "setup_set" then
    if b.setup_mode then
      b:set_piece(req.x, req.y, req.color or 0)
    end
    return get_full_state()

  elseif cmd == "setup_clear" then
    b:clear_board()
    return get_full_state()

  elseif cmd == "setup_startpos" then
    b:reset()
    b.setup_mode = true
    return get_full_state()

  elseif cmd == "setup_invert" then
    b:invert_board()
    return get_full_state()

  elseif cmd == "setup_turn" then
    b.turn = req.turn or board.BLACK
    return get_full_state()

  elseif cmd == "load_fen" then
    local ok, err = b:load_fen(req.fen, req.stm)
    if ok then
      state.eng:sync_position(b:to_uci_moves(), req.fen, req.stm)
      state.status_msg = "Loaded FEN"
    else
      state.status_msg = "FEN error: " .. tostring(err)
    end
    return get_full_state()
  end

  return get_full_state()
end

-- If executed directly, run standard I/O loop
if not ... or ... == "gui.bridge" then
  -- Expose process_message for embedding
  return {
    handle_cmd = handle_cmd,
    get_state = get_full_state,
    engine = state.eng
  }
end

-- Stdio loop
while true do
  local line = io.read("*l")
  if not line or line == "quit" then break end
  if #line > 0 then
    local req = json.decode(line)
    if req then
      state.eng:update()
      local res = handle_cmd(req)
      print(json.encode(res))
      io.stdout:flush()
    end
  end
end

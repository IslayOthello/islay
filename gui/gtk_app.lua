-- Islay Othello / Reversi — Pure Lua GTK3 Desktop Studio (WZebra + AlphaGo Edition)
-- Uses lgi (GTK3 + Cairo + GLib) with LuaJIT FFI UCI IPC to ./build/islay

local lgi     = require("lgi")
local Gtk     = lgi.require("Gtk", "3.0")
local Gdk     = lgi.require("Gdk", "3.0")
local GLib    = lgi.require("GLib")
local GObject = lgi.require("GObject")
local cairo   = lgi.cairo

local board    = require("gui.board")
local engine   = require("gui.engine")
local openings = require("gui.openings")

local app = {
  board = board.new("Othello"),
  engine = engine.new(),

  -- Play Modes: 1 = Human(B) vs AI(W), 2 = Human(W) vs AI(B), 3 = 2P, 4 = AI vs AI, 5 = Observer
  mode = 1,
  budgets = {
    { label = "1K nodes",   nodes = 1000 },
    { label = "10K nodes",  nodes = 10000 },
    { label = "50K nodes",  nodes = 50000 },
    { label = "500 ms",     movetime = 500 },
    { label = "2000 ms",    movetime = 2000 },
    { label = "Infinite",   infinite = true },
  },
  selected_budget = 2,

  -- Engine & Observer Settings
  engine_opts = {
    eval_file = "",
    mcts_hash = 512,
    mcts_batch = 12,
    q_aware_stop = true,
    perft_hash = 512,
    auto_eval_nodes = 800,
    observer_pace_ms = 600,
  },
  observer_paused = false,

  -- View Toggles (WZebra + AlphaGo)
  show_numbers = true,
  show_mcts_overlay = true,
  show_graph = true,
  show_pv_hover = true,

  -- Live State
  analyzing = false,
  auto_eval = true,
  eval_requested = false,
  hint_requested = false,
  stop_only = false,
  suppress_analyze_toggle = false,
  hint_move = nil,
  searched_turn = board.BLACK,
  black_q = 0.0,
  displayed_wr = 50.0,
  search_gen = 0,
  ai_delay = 0,
  hovered_cell = nil,
  graph_hover_ply = nil,
  setup_tool = 1, -- 1: Black, 2: White, 0: Empty
  status_text = "Initializing engine...",

  -- Widgets
  win = nil,
  board_da = nil,
  gauge_da = nil,
  graph_da = nil,
  lbl_opening = nil,
  lbl_ply = nil,
  lbl_black_score = nil,
  lbl_white_score = nil,
  lbl_black_sub = nil,
  lbl_white_sub = nil,
  lbl_black_turn = nil,
  lbl_white_turn = nil,
  lbl_nodes = nil,
  lbl_nps = nil,
  lbl_time = nil,
  lbl_winrate = nil,
  lbl_qval = nil,
  lbl_eval_name = nil,
  lbl_pv = nil,
  lbl_statusbar = nil,
  lbl_engine_pill = nil,
  moves_store = nil,
  moves_view = nil,
  cand_store = nil,
  notebook = nil,
  btn_analyze = nil,
  btn_stop = nil,
  btn_setup_turn = nil,
  observer_bar = nil,
  btn_obs_pause = nil,
  lbl_obs_status = nil,
  curve_frame = nil,
  mode_combo = nil,
  rule_combo = nil,
  budget_combo = nil,
  suppress_combos = false,
}

-- ---------------------------------------------------------------------------
-- Helpers: Winrate / Value Conversion & Formatting
-- ---------------------------------------------------------------------------
local function q_to_winrate(q)
  local wr = ((q or 0.0) + 1.0) * 50.0
  if wr < 0 then wr = 0 elseif wr > 100 then wr = 100 end
  return wr
end

-- Convert side-to-move Q to Black-perspective Q
local function to_black_q(q, stm)
  if stm == board.WHITE then return -(q or 0.0) end
  return q or 0.0
end

local function fmt_nodes(n)
  n = tonumber(n) or 0
  if n >= 1000000 then return string.format("%.2fM", n / 1000000) end
  if n >= 1000 then return string.format("%.1fk", n / 1000) end
  return tostring(n)
end

local function fmt_int(n)
  local s = tostring(math.floor(tonumber(n) or 0))
  return s:reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", "")
end

function app.is_ai_turn()
  if app.board.game_over or app.board.setup_mode then return false end
  local t = app.board.turn
  if app.mode == 1 and t == board.WHITE then return true end
  if app.mode == 2 and t == board.BLACK then return true end
  if app.mode == 4 then return true end
  if app.mode == 5 and not app.observer_paused then return true end
  return false
end

function app.set_status(msg)
  app.status_text = msg or ""
  if app.lbl_statusbar then
    app.lbl_statusbar:set_text(app.status_text)
  end
end

-- ---------------------------------------------------------------------------
-- Engine Control & Sync
-- ---------------------------------------------------------------------------
function app.sync_engine()
  local fen, stm = app.board:to_fen()
  if app.board.initial_fen and not app.board.setup_mode then
    app.engine:sync_position(app.board:to_uci_moves(), app.board.initial_fen, app.board.initial_stm)
  elseif #app.board.history > 0 and not app.board.setup_mode then
    app.engine:sync_position(app.board:to_uci_moves())
  elseif fen == "---------------------------OX------XO---------------------------" and stm == "X" then
    app.engine:sync_position("")
  else
    app.engine:sync_position("", fen, stm)
  end
end

function app.trigger_ai()
  if app.board.game_over or app.board.setup_mode then return end
  if app.analyzing then
    app.analyzing = false
    app.engine:stop()
    if app.btn_analyze then
      app.suppress_analyze_toggle = true
      app.btn_analyze:set_active(false)
      app.suppress_analyze_toggle = false
    end
  elseif app.engine.state == "searching" then
    app.engine:stop()
  end
  app.search_gen = app.search_gen + 1
  app.hint_requested = false
  app.eval_requested = false
  app.stop_only = false
  app.searched_turn = app.board.turn
  app.sync_engine()
  local bgt = app.budgets[app.selected_budget]
  if bgt.infinite then
    app.engine:search_infinite()
    app.set_status("Islay is thinking (go infinite — click Stop to play)...")
  elseif bgt.nodes then
    app.engine:search_nodes(bgt.nodes)
    app.set_status("Islay is thinking (" .. bgt.label .. ")...")
  else
    app.engine:search_movetime(bgt.movetime)
    app.set_status("Islay is thinking (" .. bgt.label .. ")...")
  end
  app.refresh_ui()
end

function app.can_show_ai_moves()
  if app.mode == 3 or app.mode == 5 then
    return app.analyzing
  end
  return true
end

function app.trigger_auto_eval()
  if not app.auto_eval or app.analyzing or app.board.game_over or app.board.setup_mode then return end
  if app.is_ai_turn() or app.mode == 3 or app.mode == 4 or app.mode == 5 then return end
  if app.engine.state ~= "ready" then return end
  app.search_gen = app.search_gen + 1
  app.hint_requested = false
  app.eval_requested = true
  app.stop_only = false
  app.searched_turn = app.board.turn
  app.sync_engine()
  app.engine:search_nodes(app.engine_opts.auto_eval_nodes or 800)
  app.refresh_ui()
end

function app.toggle_analyze()
  if app.suppress_analyze_toggle then return end
  if app.board.game_over or app.board.setup_mode then
    app.analyzing = false
    if app.btn_analyze then
      app.suppress_analyze_toggle = true
      app.btn_analyze:set_active(false)
      app.suppress_analyze_toggle = false
    end
    return
  end
  if app.analyzing then
    app.analyzing = false
    app.stop_only = true
    app.engine:stop()
    app.set_status("Stopped infinite analysis")
  else
    if app.engine.state == "searching" then
      app.engine:stop()
    end
    app.search_gen = app.search_gen + 1
    app.analyzing = true
    app.hint_requested = false
    app.eval_requested = false
    app.stop_only = false
    app.searched_turn = app.board.turn
    app.sync_engine()
    app.engine:search_infinite()
    app.set_status("Go infinite analysis running (click Stop to halt)...")
  end
  app.refresh_ui()
end

function app.stop_search()
  if app.analyzing then
    app.analyzing = false
    app.stop_only = true
    if app.btn_analyze then
      app.suppress_analyze_toggle = true
      app.btn_analyze:set_active(false)
      app.suppress_analyze_toggle = false
    end
    app.engine:stop()
    app.set_status("Stopping go infinite...")
  elseif app.engine.state == "searching" then
    app.engine:stop()
    app.set_status("Stopping search...")
  end
  app.refresh_ui()
end

function app.restart_analyze_if_active()
  if app.analyzing and not app.board.game_over and not app.board.setup_mode then
    app.engine:stop()
    app.search_gen = app.search_gen + 1
    app.searched_turn = app.board.turn
    app.sync_engine()
    app.engine:search_infinite()
  elseif app.auto_eval and not app.is_ai_turn() and not app.board.game_over and not app.board.setup_mode then
    app.ai_delay = 0.05
  end
end

function app.request_hint()
  if app.board.game_over or app.board.setup_mode then return end
  if app.analyzing then
    app.analyzing = false
    app.engine:stop()
    if app.btn_analyze then
      app.suppress_analyze_toggle = true
      app.btn_analyze:set_active(false)
      app.suppress_analyze_toggle = false
    end
  elseif app.engine.state == "searching" then
    app.engine:stop()
  end
  app.search_gen = app.search_gen + 1
  app.hint_requested = true
  app.eval_requested = false
  app.stop_only = false
  app.searched_turn = app.board.turn
  app.sync_engine()
  local bgt = app.budgets[app.selected_budget]
  if bgt.infinite then
    app.engine:search_infinite()
    app.set_status("Calculating best move (go infinite — click Stop to finish)...")
  elseif bgt.nodes then
    app.engine:search_nodes(bgt.nodes)
    app.set_status("Calculating best move & MCTS candidates...")
  else
    app.engine:search_movetime(bgt.movetime)
    app.set_status("Calculating best move & MCTS candidates...")
  end
  app.refresh_ui()
end

-- ---------------------------------------------------------------------------
-- Game Actions
-- ---------------------------------------------------------------------------
function app.cancel_active_search()
  if app.engine.state == "searching" or (app.engine.pending_searches or 0) > 0 then
    app.engine:stop()
  end
  app.eval_requested = false
  app.hint_requested = false
  app.search_gen = app.search_gen + 1
end

function app.new_game()
  if app.analyzing then
    app.analyzing = false
    if app.btn_analyze then app.btn_analyze:set_active(false) end
  end
  app.cancel_active_search()
  app.board:reset()
  app.hint_move = nil
  app.black_q = 0.0
  app.engine.stats.candidates = {}
  app.engine.stats.pv = ""
  app.sync_engine()
  app.set_status("New game started")
  if app.is_ai_turn() then
    app.ai_delay = 0.25
  elseif app.auto_eval then
    app.ai_delay = 0.15
  end
  app.refresh_ui()
end

function app.play_human_move(x, y)
  if app.board.setup_mode then
    app.board:set_piece(x, y, app.setup_tool)
    app.refresh_ui()
    return
  end
  if app.board.game_over or app.is_ai_turn() then return end

  local ok, err = app.board:apply_move(x, y, app.black_q)
  if ok then
    app.cancel_active_search()
    app.hint_move = nil
    app.engine.stats.candidates = {}
    app.engine.stats.pv = ""
    app.sync_engine()
    app.set_status("Played " .. board.to_uci(x, y):upper())
    if app.is_ai_turn() then
      app.ai_delay = 0.20
    elseif app.board.forced_pass then
      app.set_status("No legal moves — auto-passing...")
      app.ai_delay = 0.35
    else
      app.restart_analyze_if_active()
    end
  else
    app.set_status("Illegal move: " .. tostring(err))
  end
  app.refresh_ui()
end

function app.do_pass()
  if app.board.game_over or app.board.setup_mode then return end
  local ok, err = app.board:pass(app.black_q)
  if ok then
    app.cancel_active_search()
    app.hint_move = nil
    app.engine.stats.candidates = {}
    app.sync_engine()
    app.set_status("Passed turn")
    if app.is_ai_turn() then
      app.ai_delay = 0.20
    elseif app.board.forced_pass then
      app.set_status("No legal moves — auto-passing...")
      app.ai_delay = 0.35
    else
      app.restart_analyze_if_active()
    end
  else
    app.set_status(tostring(err or "Cannot pass"))
  end
  app.refresh_ui()
end

function app.undo()
  if app.board:undo() then
    app.cancel_active_search()
    app.hint_move = nil
    app.engine.stats.candidates = {}
    app.sync_engine()
    app.restart_analyze_if_active()
    app.set_status("Took back move (Ply " .. app.board.current_ply .. ")")
    app.refresh_ui()
  end
end

function app.redo()
  if app.board:redo() then
    app.cancel_active_search()
    app.hint_move = nil
    app.engine.stats.candidates = {}
    app.sync_engine()
    app.restart_analyze_if_active()
    app.set_status("Stepped forward (Ply " .. app.board.current_ply .. ")")
    app.refresh_ui()
  end
end

function app.goto_ply(ply)
  app.cancel_active_search()
  app.board:goto_ply(ply)
  app.hint_move = nil
  app.engine.stats.candidates = {}
  app.sync_engine()
  app.restart_analyze_if_active()
  app.set_status("Viewing ply " .. app.board.current_ply)
  app.refresh_ui()
end

function app.set_mode(m)
  app.mode = m
  app.observer_paused = false
  app.suppress_combos = true
  if app.mode_combo then app.mode_combo:set_active(m - 1) end
  app.suppress_combos = false
  if (m == 3 or m == 5) and not app.analyzing then
    app.cancel_active_search()
    app.hint_move = nil
    app.engine.stats.candidates = {}
    app.engine.stats.pv = ""
  end
  if app.is_ai_turn() and not app.board.game_over then
    app.ai_delay = (m == 5) and ((app.engine_opts.observer_pace_ms or 600) / 1000.0) or 0.25
  elseif app.auto_eval and not app.is_ai_turn() and m ~= 3 and m ~= 5 and not app.board.game_over then
    app.ai_delay = 0.10
  end
  app.refresh_ui()
end

function app.observer_toggle_pause()
  app.observer_paused = not app.observer_paused
  if app.observer_paused then
    app.cancel_active_search()
    app.set_status("Observer paused at Move " .. app.board.current_ply)
  else
    app.set_status("Observer resumed")
    if app.is_ai_turn() and not app.board.game_over then
      app.ai_delay = 0.15
    end
  end
  app.refresh_ui()
end

function app.observer_step_once()
  if app.board.game_over or app.board.setup_mode then return end
  app.observer_paused = true
  app.trigger_ai()
end

function app.load_eval_file_dialog(parent_win)
  local fc = Gtk.FileChooserDialog {
    title = "Load Eval File (.onnx / .nnue / .pat / .bin)",
    transient_for = parent_win or app.win,
    action = Gtk.FileChooserAction.OPEN
  }
  fc:add_button("Cancel", Gtk.ResponseType.CANCEL)
  fc:add_button("Open", Gtk.ResponseType.ACCEPT)

  local f_eval = Gtk.FileFilter()
  f_eval:set_name("Eval Files (*.onnx, *.nnue, *.pat, *.bin)")
  f_eval:add_pattern("*.onnx")
  f_eval:add_pattern("*.nnue")
  f_eval:add_pattern("*.pat")
  f_eval:add_pattern("*.bin")
  fc:add_filter(f_eval)

  local f_all = Gtk.FileFilter()
  f_all:set_name("All Files (*)")
  f_all:add_pattern("*")
  fc:add_filter(f_all)

  local chosen = nil
  if fc:run() == Gtk.ResponseType.ACCEPT then
    chosen = fc:get_filename()
  end
  fc:destroy()
  return chosen
end

function app.load_eval_file(path)
  if not path then return end
  local was_analyzing = app.analyzing
  app.cancel_active_search()
  app.engine.stats.candidates = {}
  app.engine.stats.nodes = 0
  app.engine_opts.eval_file = path
  app.engine:set_eval_file(path)
  app.engine:send("isready")
  local label = (path == "") and "Uniform (No Network)" or (path:match("([^/\\]+)$") or path)
  app.set_status("Loaded EvalFile: " .. label)
  if was_analyzing then
    app.analyzing = true
    app.restart_analyze_if_active()
  elseif app.auto_eval and not app.is_ai_turn() and app.mode ~= 3 and app.mode ~= 5 and not app.board.game_over then
    app.ai_delay = 0.12
  end
  app.refresh_ui()
end

function app.apply_engine_options(opts)
  local was_analyzing = app.analyzing
  app.cancel_active_search()
  app.engine.stats.candidates = {}
  app.engine.stats.nodes = 0
  app.engine_opts.eval_file = opts.eval_file or ""
  app.engine_opts.mcts_hash = tonumber(opts.mcts_hash) or 512
  app.engine_opts.mcts_batch = tonumber(opts.mcts_batch) or 12
  app.engine_opts.q_aware_stop = (opts.q_aware_stop == true)
  app.engine_opts.perft_hash = tonumber(opts.perft_hash) or 512
  app.engine_opts.auto_eval_nodes = tonumber(opts.auto_eval_nodes) or 800
  app.engine_opts.observer_pace_ms = tonumber(opts.observer_pace_ms) or 600

  app.engine:set_eval_file(app.engine_opts.eval_file)
  app.engine:set_mcts_hash(app.engine_opts.mcts_hash)
  app.engine:set_mcts_batch(app.engine_opts.mcts_batch)
  app.engine:set_q_aware_stop(app.engine_opts.q_aware_stop)
  app.engine:set_perft_hash(app.engine_opts.perft_hash)
  app.engine:send("isready")
  app.set_status("Applied engine settings")
  if was_analyzing then
    app.analyzing = true
    app.restart_analyze_if_active()
  elseif app.auto_eval and not app.is_ai_turn() and app.mode ~= 3 and app.mode ~= 5 and not app.board.game_over then
    app.ai_delay = 0.12
  end
  app.refresh_ui()
end

function app.open_engine_settings()
  local dlg = Gtk.Dialog {
    title = "Engine Settings",
    transient_for = app.win,
    modal = true,
    default_width = 460
  }
  dlg:add_button("Cancel", Gtk.ResponseType.CANCEL)
  dlg:add_button("Apply", Gtk.ResponseType.APPLY)
  dlg:add_button("OK", Gtk.ResponseType.OK)

  local content = dlg:get_content_area()
  local grid = Gtk.Grid {
    column_spacing = 12,
    row_spacing = 10,
    margin = 14
  }
  content:pack_start(grid, true, true, 0)

  local function add_lbl(row, text)
    local l = Gtk.Label { label = text, xalign = 0 }
    grid:attach(l, 0, row, 1, 1)
  end

  -- Row 0: EvalFile preset + entry + Browse
  add_lbl(0, "EvalFile (ONNX / NNUE):")
  local eval_box = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 4 }
  local preset_combo = Gtk.ComboBoxText()
  local presets = {
    { label = "Custom / Current", path = nil },
    { label = "wdl-gen5.onnx", path = "./wdl-gen5.onnx" },
    { label = "wdl-gen4.onnx", path = "./wdl-gen4.onnx" },
    { label = "wdl-gen3.onnx", path = "./wdl-gen3.onnx" },
    { label = "wdl-gen2.onnx", path = "./wdl-gen2.onnx" },
    { label = "wdl-gen1.onnx", path = "./wdl-gen1.onnx" },
    { label = "Uniform (No Network)", path = "" },
  }
  for _, p in ipairs(presets) do preset_combo:append_text(p.label) end
  preset_combo:set_active(0)

  local eval_row = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 6 }
  local cur_eval = app.engine_opts.eval_file
  if cur_eval == "" and app.engine.eval_file then cur_eval = app.engine.eval_file end
  local ent_eval = Gtk.Entry { text = cur_eval or "", hexpand = true }
  local btn_browse = Gtk.Button.new_with_label("Browse...")
  btn_browse.on_clicked = function()
    local fn = app.load_eval_file_dialog(dlg)
    if fn then ent_eval:set_text(fn) end
  end
  preset_combo.on_changed = function()
    local idx = preset_combo:get_active() + 1
    if presets[idx] and presets[idx].path ~= nil then
      ent_eval:set_text(presets[idx].path)
    end
  end
  eval_row:pack_start(ent_eval, true, true, 0)
  eval_row:pack_end(btn_browse, false, false, 0)
  eval_box:pack_start(preset_combo, false, false, 0)
  eval_box:pack_start(eval_row, false, false, 0)
  grid:attach(eval_box, 1, 0, 1, 1)

  -- Row 1: MctsHash
  add_lbl(1, "MCTS Hash (MiB):")
  local sp_mhash = Gtk.SpinButton.new_with_range(1, 4096, 16)
  sp_mhash:set_value(app.engine_opts.mcts_hash or 512)
  grid:attach(sp_mhash, 1, 1, 1, 1)

  -- Row 2: MctsBatch
  add_lbl(2, "MCTS Batch Size:")
  local sp_batch = Gtk.SpinButton.new_with_range(1, 16, 1)
  sp_batch:set_value(app.engine_opts.mcts_batch or 12)
  grid:attach(sp_batch, 1, 2, 1, 1)

  -- Row 3: MctsQAwareStop
  add_lbl(3, "Q-Aware Early Stop:")
  local chk_qstop = Gtk.CheckButton.new_with_label("Stop early when top move Q gap >= 0.05")
  chk_qstop:set_active(app.engine_opts.q_aware_stop == true)
  grid:attach(chk_qstop, 1, 3, 1, 1)

  -- Row 4: PerftHash
  add_lbl(4, "Perft Hash (MiB):")
  local sp_phash = Gtk.SpinButton.new_with_range(1, 65536, 64)
  sp_phash:set_value(app.engine_opts.perft_hash or 512)
  grid:attach(sp_phash, 1, 4, 1, 1)

  -- Row 5: Auto Eval Nodes
  add_lbl(5, "Auto Eval Nodes (Human turn):")
  local sp_autonodes = Gtk.SpinButton.new_with_range(100, 10000, 100)
  sp_autonodes:set_value(app.engine_opts.auto_eval_nodes or 800)
  grid:attach(sp_autonodes, 1, 5, 1, 1)

  -- Row 6: Observer Pace (ms)
  add_lbl(6, "Observer Pace (ms / move):")
  local sp_pace = Gtk.SpinButton.new_with_range(100, 5000, 100)
  sp_pace:set_value(app.engine_opts.observer_pace_ms or 600)
  grid:attach(sp_pace, 1, 6, 1, 1)

  dlg:show_all()

  local function read_form()
    return {
      eval_file = ent_eval:get_text(),
      mcts_hash = sp_mhash:get_value_as_int(),
      mcts_batch = sp_batch:get_value_as_int(),
      q_aware_stop = chk_qstop:get_active(),
      perft_hash = sp_phash:get_value_as_int(),
      auto_eval_nodes = sp_autonodes:get_value_as_int(),
      observer_pace_ms = sp_pace:get_value_as_int(),
    }
  end

  while true do
    local resp = dlg:run()
    if resp == Gtk.ResponseType.APPLY then
      app.apply_engine_options(read_form())
    elseif resp == Gtk.ResponseType.OK then
      app.apply_engine_options(read_form())
      break
    else
      break
    end
  end
  dlg:destroy()
end

function app.set_rule(rule)
  app.board.rule = rule
  app.board:check_game_state()
  app.engine:set_rule(rule)
  app.suppress_combos = true
  if app.rule_combo then app.rule_combo:set_active(rule == "Reversi" and 1 or 0) end
  app.suppress_combos = false
  app.set_status("Rule set to " .. rule)
  app.refresh_ui()
end

function app.copy_fen()
  local fen, stm = app.board:to_fen()
  local full = fen .. " " .. stm
  local cb = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
  cb:set_text(full, -1)
  app.set_status("Copied FEN to clipboard")
end

function app.paste_fen()
  local cb = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
  local txt = cb:wait_for_text()
  if not txt or txt == "" then
    app.set_status("Clipboard is empty")
    return
  end
  local fen, stm = txt:match("^%s*([%-%w]+)%s+([XOoxBWbw]?)")
  if not fen then fen = txt:gsub("%s+", "") end
  stm = (stm and stm ~= "") and stm:upper() or "X"
  local ok, err = app.board:load_fen(fen, stm)
  if ok then
    app.hint_move = nil
    app.engine.stats.candidates = {}
    app.sync_engine()
    app.set_status("Loaded FEN position")
    app.refresh_ui()
  else
    app.set_status("Invalid FEN: " .. tostring(err))
  end
end

function app.run_perft(depth)
  app.sync_engine()
  app.engine:perft(depth or 5)
  app.set_status("Running Perft depth " .. tostring(depth or 5) .. "...")
end

-- ---------------------------------------------------------------------------
-- Cairo Drawing: 8x8 Board (WZebra Felt + AlphaGo MCTS Overlay + PV Hover)
-- ---------------------------------------------------------------------------
local function board_geometry(w, h)
  local size = math.min(w, h)
  local margin = math.max(22, math.floor(size * 0.045))
  local grid = size - margin * 2
  local cs = grid / 8
  local ox = (w - size) / 2
  local oy = (h - size) / 2
  return ox, oy, size, margin, grid, cs
end

local function draw_disc(cr, cx, cy, r, color, move_num, is_last, alpha)
  alpha = alpha or 1.0
  cr:save()

  -- Drop shadow
  cr:arc(cx + 1.5, cy + 2.5, r, 0, math.pi * 2)
  cr:set_source_rgba(0, 0, 0, 0.32 * alpha)
  cr:fill()

  -- Disc body gradient
  local pat = cairo.Pattern.create_radial(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r)
  if color == board.BLACK then
    pat:add_color_stop_rgba(0.0, 0.28, 0.30, 0.34, alpha)
    pat:add_color_stop_rgba(0.55, 0.11, 0.12, 0.14, alpha)
    pat:add_color_stop_rgba(1.0, 0.03, 0.03, 0.04, alpha)
  else
    pat:add_color_stop_rgba(0.0, 1.00, 1.00, 1.00, alpha)
    pat:add_color_stop_rgba(0.70, 0.94, 0.95, 0.97, alpha)
    pat:add_color_stop_rgba(1.0, 0.78, 0.81, 0.86, alpha)
  end
  cr:arc(cx, cy, r, 0, math.pi * 2)
  cr:set_source(pat)
  cr:fill()

  -- Rim stroke
  cr:arc(cx, cy, r, 0, math.pi * 2)
  if color == board.BLACK then
    cr:set_source_rgba(0.02, 0.02, 0.03, alpha)
  else
    cr:set_source_rgba(0.68, 0.72, 0.78, alpha)
  end
  cr:set_line_width(1.2)
  cr:stroke()

  -- Last move indicator
  if is_last then
    cr:arc(cx, cy, r * 0.86, 0, math.pi * 2)
    cr:set_source_rgba(0.98, 0.74, 0.18, 0.95 * alpha)
    cr:set_line_width(2.2)
    cr:stroke()
  end

  -- Move number (WZebra style)
  if move_num and move_num > 0 then
    cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
    cr:set_font_size(math.max(10, r * 0.56))
    local txt = tostring(move_num)
    local ext = cr:text_extents(txt)
    if color == board.BLACK then
      cr:set_source_rgba(0.95, 0.96, 0.98, 0.92 * alpha)
    else
      cr:set_source_rgba(0.12, 0.14, 0.18, 0.92 * alpha)
    end
    cr:move_to(cx - ext.width / 2 - ext.x_bearing, cy - ext.height / 2 - ext.y_bearing)
    cr:show_text(txt)
  end

  cr:restore()
end

local function on_draw_board(widget, cr)
  local alloc = widget:get_allocation()
  local ox, oy, size, margin, grid, cs = board_geometry(alloc.width, alloc.height)

  -- 1. Outer Bezel (Satin dark wood/slate)
  cr:set_source_rgb(0.10, 0.13, 0.11)
  cr:rectangle(ox, oy, size, size)
  cr:fill()

  -- Coordinates A-H, 1-8
  cr:select_font_face("Monospace", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
  cr:set_font_size(math.max(10, margin * 0.48))
  cr:set_source_rgb(0.75, 0.80, 0.76)
  for i = 0, 7 do
    local col = string.char(65 + i)
    local row = tostring(i + 1)
    local c_pos = ox + margin + (i + 0.5) * cs
    local r_pos = oy + margin + (i + 0.5) * cs

    local ce = cr:text_extents(col)
    cr:move_to(c_pos - ce.width / 2 - ce.x_bearing, oy + margin * 0.65)
    cr:show_text(col)
    cr:move_to(c_pos - ce.width / 2 - ce.x_bearing, oy + size - margin * 0.25)
    cr:show_text(col)

    local re = cr:text_extents(row)
    cr:move_to(ox + margin * 0.32, r_pos - re.height / 2 - re.y_bearing)
    cr:show_text(row)
    cr:move_to(ox + size - margin * 0.68, r_pos - re.height / 2 - re.y_bearing)
    cr:show_text(row)
  end

  -- 2. Emerald Tournament Felt
  local gx, gy = ox + margin, oy + margin
  cr:set_source_rgb(0.105, 0.41, 0.22)
  cr:rectangle(gx, gy, grid, grid)
  cr:fill()

  -- Subtle alternating squares
  cr:set_source_rgba(0, 0, 0, 0.04)
  for y = 0, 7 do
    for x = 0, 7 do
      if (x + y) % 2 == 1 then
        cr:rectangle(gx + x * cs, gy + y * cs, cs, cs)
        cr:fill()
      end
    end
  end

  -- Grid lines
  cr:set_source_rgb(0.05, 0.24, 0.12)
  cr:set_line_width(1.1)
  for i = 0, 8 do
    cr:move_to(gx + i * cs, gy)
    cr:line_to(gx + i * cs, gy + grid)
    cr:move_to(gx, gy + i * cs)
    cr:line_to(gx + grid, gy + i * cs)
  end
  cr:stroke()

  -- 4 Star points (c3, g3, c7, g7 intersections -> 2, 6)
  cr:set_source_rgb(0.05, 0.22, 0.11)
  for _, sx in ipairs({ 2, 6 }) do
    for _, sy in ipairs({ 2, 6 }) do
      cr:arc(gx + sx * cs, gy + sy * cs, 3.6, 0, math.pi * 2)
      cr:fill()
    end
  end

  -- 3. Hover & Flip Preview
  local legal_moves = (not app.board.game_over and not app.board.setup_mode)
                      and app.board:get_legal_moves(app.board.turn) or {}
  local hover_flips = {}
  local hovered_uci = nil
  if app.hovered_cell then
    local hx, hy = app.hovered_cell.x, app.hovered_cell.y
    hovered_uci = board.to_uci(hx, hy)
    if legal_moves[hovered_uci] then
      cr:set_source_rgba(1, 1, 1, 0.14)
      cr:rectangle(gx + (hx - 1) * cs + 1, gy + (hy - 1) * cs + 1, cs - 2, cs - 2)
      cr:fill()
      for _, f in ipairs(legal_moves[hovered_uci].flips) do
        hover_flips[f.y .. "_" .. f.x] = true
      end
    elseif app.board.setup_mode then
      cr:set_source_rgba(0.2, 0.6, 1.0, 0.22)
      cr:rectangle(gx + (hx - 1) * cs + 1, gy + (hy - 1) * cs + 1, cs - 2, cs - 2)
      cr:fill()
    end
  end

  -- 4. Discs
  local disc_r = cs * 0.40
  for y = 1, 8 do
    for x = 1, 8 do
      local cell = app.board.grid[y][x]
      if cell ~= board.EMPTY then
        local cx = gx + (x - 0.5) * cs
        local cy = gy + (y - 0.5) * cs
        local is_last = app.board.last_move and app.board.last_move.x == x and app.board.last_move.y == y
        local mnum = app.show_numbers and app.board:get_move_number(x, y) or 0
        draw_disc(cr, cx, cy, disc_r, cell, mnum, is_last, 1.0)

        -- Highlight discs that will flip on hover
        if hover_flips[y .. "_" .. x] then
          cr:arc(cx, cy, disc_r * 0.90, 0, math.pi * 2)
          cr:set_source_rgba(0.20, 0.78, 0.98, 0.85)
          cr:set_line_width(2.2)
          cr:stroke()
        end
      end
    end
  end

  -- 5. Legal Move Markers & AlphaGo/Lizzie MCTS Candidate Circles
  local show_ai_moves = app.can_show_ai_moves()
  local cand_by_uci = {}
  local best_wr = -1
  if show_ai_moves and app.show_mcts_overlay and app.engine.stats.candidates then
    for idx, c in ipairs(app.engine.stats.candidates) do
      if (c.visits or 0) > 0 then
        local wr = q_to_winrate(c.value)
        if idx == 1 or wr > best_wr then best_wr = wr end
        cand_by_uci[c.move] = { rank = idx, data = c, wr = wr }
      end
    end
  end

  for uci, m in pairs(legal_moves) do
    local cx = gx + (m.x - 0.5) * cs
    local cy = gy + (m.y - 0.5) * cs
    local cand = cand_by_uci[uci]
    local is_hint = show_ai_moves and (app.hint_move == uci)

    if cand then
      local cr_rad = cs * 0.38
      local delta = best_wr - cand.wr
      local r_col, g_col, b_col
      if cand.rank == 1 or is_hint then
        r_col, g_col, b_col = 0.04, 0.58, 0.95 -- AlphaGo Blue (Best Move)
      elseif delta <= 3.0 then
        r_col, g_col, b_col = 0.12, 0.74, 0.42 -- Green (Strong)
      elseif delta <= 8.0 then
        r_col, g_col, b_col = 0.92, 0.62, 0.10 -- Amber
      else
        r_col, g_col, b_col = 0.86, 0.25, 0.25 -- Red
      end

      -- Translucent circle badge
      cr:arc(cx, cy, cr_rad, 0, math.pi * 2)
      cr:set_source_rgba(r_col, g_col, b_col, 0.82)
      cr:fill()
      cr:arc(cx, cy, cr_rad, 0, math.pi * 2)
      cr:set_source_rgba(1, 1, 1, 0.85)
      cr:set_line_width(cand.rank == 1 and 2.2 or 1.2)
      cr:stroke()

      -- Winrate % (top) & Visits (bottom)
      cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
      cr:set_font_size(math.max(9, cs * 0.21))
      local wr_txt = string.format("%.0f%%", cand.wr)
      local we = cr:text_extents(wr_txt)
      cr:set_source_rgb(1, 1, 1)
      cr:move_to(cx - we.width / 2 - we.x_bearing, cy - cs * 0.04)
      cr:show_text(wr_txt)

      cr:set_font_size(math.max(8, cs * 0.16))
      local vis_txt = fmt_nodes(cand.data.visits)
      local ve = cr:text_extents(vis_txt)
      cr:set_source_rgba(1, 1, 1, 0.90)
      cr:move_to(cx - ve.width / 2 - ve.x_bearing, cy + cs * 0.20)
      cr:show_text(vis_txt)
    else
      -- Standard legal dot
      cr:arc(cx, cy, is_hint and cs * 0.22 or cs * 0.11, 0, math.pi * 2)
      if is_hint then
        cr:set_source_rgba(0.05, 0.72, 0.95, 0.92)
      elseif hovered_uci == uci then
        cr:set_source_rgba(0.98, 0.76, 0.20, 0.95)
      else
        cr:set_source_rgba(1, 1, 1, 0.45)
      end
      cr:fill()
    end

    -- Flip count badge on hover
    if hovered_uci == uci and #m.flips > 0 and not cand then
      cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
      cr:set_font_size(math.max(9, cs * 0.19))
      local ftxt = "+" .. tostring(#m.flips)
      local fe = cr:text_extents(ftxt)
      cr:set_source_rgba(1, 0.95, 0.75, 0.95)
      cr:move_to(cx - fe.width / 2 - fe.x_bearing, cy - cs * 0.24)
      cr:show_text(ftxt)
    end
  end

  -- 6. AlphaGo PV Preview on Hover (when hovering the PV's root move)
  if show_ai_moves and app.show_pv_hover and hovered_uci and app.engine.stats.pv and app.engine.stats.pv ~= "" then
    local pv_moves = {}
    for tok in app.engine.stats.pv:gmatch("%S+") do
      table.insert(pv_moves, tok)
    end
    if pv_moves[1] == hovered_uci then
      local sim_turn = app.board.turn
      for step = 1, math.min(#pv_moves, 6) do
        local mv = pv_moves[step]
        if mv ~= "pass" then
          local px, py = board.from_uci(mv)
          if px and py and app.board.grid[py][px] == board.EMPTY then
            local cx = gx + (px - 0.5) * cs
            local cy = gy + (py - 0.5) * cs
            draw_disc(cr, cx, cy, disc_r * 0.86, sim_turn, step, false, 0.58)
          end
        end
        sim_turn = app.board:opponent(sim_turn)
      end
    end
  end

  -- 7. Game Over / Forced Pass Banner Overlay
  if app.board.game_over then
    cr:set_source_rgba(0.05, 0.08, 0.06, 0.65)
    cr:rectangle(gx, gy + grid * 0.40, grid, grid * 0.20)
    cr:fill()
    local b_cnt, w_cnt = app.board:get_counts()
    local msg = b_cnt > w_cnt and string.format("BLACK WINS (%d – %d)", b_cnt, w_cnt)
             or w_cnt > b_cnt and string.format("WHITE WINS (%d – %d)", w_cnt, b_cnt)
             or string.format("DRAW (%d – %d)", b_cnt, w_cnt)
    cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
    cr:set_font_size(math.max(16, cs * 0.34))
    local me = cr:text_extents(msg)
    cr:set_source_rgb(1, 1, 1)
    cr:move_to(gx + grid / 2 - me.width / 2 - me.x_bearing, gy + grid / 2 - me.height / 2 - me.y_bearing)
    cr:show_text(msg)
  end
end

-- ---------------------------------------------------------------------------
-- Cairo Drawing: Horizontal Winrate Eval Bar (Top of Board)
-- ---------------------------------------------------------------------------
local function on_draw_gauge(widget, cr)
  local alloc = widget:get_allocation()
  local w, h = alloc.width, alloc.height
  local black_wr = app.displayed_wr or 50.0
  local white_wr = 100.0 - black_wr
  local is_searching = (app.engine.state == "searching")

  local pad_y = 2
  local bar_x = 0
  local bar_y = pad_y
  local bar_w = w
  local bar_h = h - pad_y * 2

  -- White right background
  cr:set_source_rgb(0.95, 0.96, 0.98)
  cr:rectangle(bar_x, bar_y, bar_w, bar_h)
  cr:fill()

  -- Black left portion
  local black_w = bar_w * (black_wr / 100.0)
  cr:set_source_rgb(0.11, 0.13, 0.17)
  cr:rectangle(bar_x, bar_y, black_w, bar_h)
  cr:fill()

  -- Border (blue highlight while engine is thinking/searching)
  if is_searching then
    cr:set_source_rgb(0.04, 0.45, 0.85)
    cr:set_line_width(1.8)
  else
    cr:set_source_rgb(0.75, 0.78, 0.82)
    cr:set_line_width(1.0)
  end
  cr:rectangle(bar_x + 0.5, bar_y + 0.5, bar_w - 1, bar_h - 1)
  cr:stroke()

  -- 50% center vertical tick
  cr:set_source_rgb(0.04, 0.45, 0.85)
  cr:set_line_width(2)
  cr:move_to(bar_x + bar_w / 2, bar_y)
  cr:line_to(bar_x + bar_w / 2, bar_y + bar_h)
  cr:stroke()

  -- Black % (left) and White % (right) labels with dual-clip high contrast
  cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
  cr:set_font_size(11)

  local b_txt = string.format("Black %.1f%%", black_wr)
  local be = cr:text_extents(b_txt)
  local disc_r = 4.2
  local bx = bar_x + 8 + disc_r * 2 + 5
  local by = bar_y + bar_h / 2 - be.height / 2 - be.y_bearing
  local b_disc_x = bar_x + 8 + disc_r
  local b_disc_y = bar_y + bar_h / 2

  local w_txt = string.format("%.1f%% White", white_wr)
  local we = cr:text_extents(w_txt)
  local w_disc_x = bar_x + bar_w - 8 - disc_r
  local w_disc_y = bar_y + bar_h / 2
  local wx = w_disc_x - disc_r - 5 - we.width - we.x_bearing
  local wy = bar_y + bar_h / 2 - we.height / 2 - we.y_bearing

  local function draw_bar_labels(text_r, text_g, text_b)
    cr:set_source_rgb(text_r, text_g, text_b)
    cr:move_to(bx, by)
    cr:show_text(b_txt)
    cr:move_to(wx, wy)
    cr:show_text(w_txt)
    -- Black mini-disc
    cr:new_sub_path()
    cr:arc(b_disc_x, b_disc_y, disc_r, 0, math.pi * 2)
    cr:set_source_rgb(0.08, 0.10, 0.14)
    cr:fill_preserve()
    cr:set_source_rgb(text_r, text_g, text_b)
    cr:set_line_width(1.1)
    cr:stroke()
    -- White mini-disc
    cr:new_sub_path()
    cr:arc(w_disc_x, w_disc_y, disc_r, 0, math.pi * 2)
    cr:set_source_rgb(0.97, 0.98, 1.0)
    cr:fill_preserve()
    cr:set_source_rgb(0.25, 0.28, 0.34)
    cr:set_line_width(1.1)
    cr:stroke()
  end

  -- Pass 1: Clip to dark (Black) region -> draw light text
  cr:save()
  cr:rectangle(bar_x, bar_y, black_w, bar_h)
  cr:clip()
  draw_bar_labels(0.96, 0.97, 0.99)
  cr:restore()

  -- Pass 2: Clip to light (White) region -> draw dark text
  cr:save()
  cr:rectangle(bar_x + black_w, bar_y, bar_w - black_w, bar_h)
  cr:clip()
  draw_bar_labels(0.11, 0.13, 0.17)
  cr:restore()
end

-- ---------------------------------------------------------------------------
-- Cairo Drawing: Winrate / Advantage Trajectory Graph (WZebra + AlphaGo)
-- ---------------------------------------------------------------------------
local function on_draw_graph(widget, cr)
  local alloc = widget:get_allocation()
  local w, h = alloc.width, alloc.height

  -- Background
  cr:set_source_rgb(1, 1, 1)
  cr:rectangle(0, 0, w, h)
  cr:fill()

  -- Horizontal grid lines (25%, 50%, 75%)
  cr:set_source_rgb(0.92, 0.94, 0.96)
  cr:set_line_width(1)
  for _, frac in ipairs({ 0.25, 0.75 }) do
    cr:move_to(0, h * frac)
    cr:line_to(w, h * frac)
  end
  cr:stroke()

  -- 50% Equal line
  cr:set_source_rgb(0.74, 0.78, 0.84)
  cr:set_dash({ 4, 3 }, 0)
  cr:move_to(0, h * 0.5)
  cr:line_to(w, h * 0.5)
  cr:stroke()
  cr:set_dash({}, 0)

  local hist = app.board.history
  if #hist == 0 then
    cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.NORMAL)
    cr:set_font_size(11)
    cr:set_source_rgb(0.55, 0.60, 0.65)
    local msg = "Play moves to plot Black Winrate trajectory (50% = Even)"
    local ext = cr:text_extents(msg)
    cr:move_to(w / 2 - ext.width / 2 - ext.x_bearing, h / 2 - ext.height / 2 - ext.y_bearing)
    cr:show_text(msg)
    return
  end

  local pad_x = 12
  local gw = w - pad_x * 2
  local total_pts = math.max(#hist, 1)

  local function pt_xy(ply_idx, q_val)
    local x = pad_x + (ply_idx / total_pts) * gw
    local wr = q_to_winrate(q_val)
    local y = (1.0 - wr / 100.0) * (h - 14) + 7
    return x, y
  end

  -- Shaded area under curve
  local q0 = (app.board.current_ply == 0 and app.engine.state == "searching") and (app.black_q or 0.0) or 0.0
  cr:move_to(pad_x, h * 0.5)
  local x0, y0 = pt_xy(0, q0)
  cr:line_to(x0, y0)
  for i = 1, #hist do
    local xi, yi = pt_xy(i, hist[i].eval_score or 0.0)
    cr:line_to(xi, yi)
  end
  local xl = pt_xy(#hist, hist[#hist].eval_score or 0.0)
  cr:line_to(xl, h * 0.5)
  cr:close_path()
  cr:set_source_rgba(0.04, 0.41, 0.85, 0.14)
  cr:fill()

  -- Curve line
  cr:move_to(x0, y0)
  for i = 1, #hist do
    local xi, yi = pt_xy(i, hist[i].eval_score or 0.0)
    cr:line_to(xi, yi)
  end
  cr:set_source_rgb(0.04, 0.41, 0.85)
  cr:set_line_width(2.0)
  cr:stroke()

  -- Current ply vertical marker + dot
  local cur_q = (app.board.current_ply > 0 and hist[app.board.current_ply])
                and (hist[app.board.current_ply].eval_score or 0.0) or q0
  local cx, cy = pt_xy(app.board.current_ply, cur_q)
  cr:set_source_rgba(0.90, 0.45, 0.05, 0.65)
  cr:set_line_width(1.2)
  cr:move_to(cx, 0)
  cr:line_to(cx, h)
  cr:stroke()

  cr:arc(cx, cy, 4.2, 0, math.pi * 2)
  cr:set_source_rgb(0.95, 0.52, 0.08)
  cr:fill()

  -- Hover Tooltip on Graph (Move, Winrate Delta, Current Winrate)
  local hp = app.graph_hover_ply
  if hp ~= nil and hp >= 0 and hp <= #hist then
    local h_q = (hp > 0 and hist[hp]) and (hist[hp].eval_score or 0.0) or 0.0
    local hx, hy = pt_xy(hp, h_q)

    -- Vertical hover hairline
    cr:set_source_rgba(0.15, 0.18, 0.22, 0.45)
    cr:set_line_width(1.0)
    cr:set_dash({ 3, 2 }, 0)
    cr:move_to(hx, 0)
    cr:line_to(hx, h)
    cr:stroke()
    cr:set_dash({}, 0)

    -- Highlighted hover point
    cr:arc(hx, hy, 4.5, 0, math.pi * 2)
    cr:set_source_rgb(0.04, 0.41, 0.85)
    cr:fill()
    cr:arc(hx, hy, 4.5, 0, math.pi * 2)
    cr:set_source_rgb(1, 1, 1)
    cr:set_line_width(1.5)
    cr:stroke()

    local tip_main, tip_delta, delta_kind
    if hp == 0 then
      tip_main = "Start · B: 50.0% / W: 50.0%"
      tip_delta = nil
    else
      local m = hist[hp]
      local wr_black = q_to_winrate(m.eval_score or 0.0)
      local prev_wr_black = (hp > 1) and q_to_winrate(hist[hp - 1].eval_score or 0.0) or 50.0
      local is_black = (m.color == board.BLACK)
      local mover_wr = is_black and wr_black or (100.0 - wr_black)
      local prev_mover_wr = is_black and prev_wr_black or (100.0 - prev_wr_black)
      local delta = mover_wr - prev_mover_wr
      if delta < -0.5 then
        delta_kind = "loss"
      elseif delta > 0.5 then
        delta_kind = "gain"
      else
        delta_kind = "flat"
      end
      local side_full = is_black and "Black" or "White"
      local side_short = is_black and "B" or "W"
      tip_main = string.format("#%d %s (%s) · %s: %.1f%%", hp, m.uci:upper(), side_full, side_short, mover_wr)
      tip_delta = string.format(" (%+.1f%%)", delta)
    end

    cr:select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
    cr:set_font_size(10.5)
    local ext1 = cr:text_extents(tip_main)
    local ext2 = tip_delta and cr:text_extents(tip_delta) or { width = 0, height = 0 }
    local box_w = ext1.width + ext2.width + 16
    local box_h = 20
    local bx = hx + 8
    if bx + box_w > w - 4 then bx = hx - box_w - 8 end
    if bx < 4 then bx = 4 end
    local by = 4

    -- Tooltip card
    cr:set_source_rgba(0.12, 0.14, 0.18, 0.92)
    cr:rectangle(bx, by, box_w, box_h)
    cr:fill()

    local ty = by + 13.5
    cr:set_source_rgb(1, 1, 1)
    cr:move_to(bx + 8, ty)
    cr:show_text(tip_main)

    if tip_delta then
      if delta_kind == "loss" then
        cr:set_source_rgb(1.0, 0.48, 0.48)
      elseif delta_kind == "gain" then
        cr:set_source_rgb(0.45, 0.92, 0.62)
      else
        cr:set_source_rgb(0.80, 0.84, 0.88)
      end
      cr:move_to(bx + 8 + ext1.width, ty)
      cr:show_text(tip_delta)
    end
  end
end

-- ---------------------------------------------------------------------------
-- Refresh Native GTK3 UI State
-- ---------------------------------------------------------------------------
function app.refresh_ui()
  local b_cnt, w_cnt = app.board:get_counts()
  app.lbl_black_score:set_text(tostring(b_cnt))
  app.lbl_white_score:set_text(tostring(w_cnt))

  local b_is_ai = (app.mode == 2 or app.mode == 4 or app.mode == 5)
  local w_is_ai = (app.mode == 1 or app.mode == 4 or app.mode == 5)
  local ai_tag = (app.mode == 5) and "islay (Observer)" or "islay MCTS"
  app.lbl_black_sub:set_text(b_is_ai and ai_tag or "Human")
  app.lbl_white_sub:set_text(w_is_ai and ai_tag or "Human")

  if app.observer_bar then
    app.observer_bar:set_visible(app.mode == 5 or app.mode == 4)
  end
  if app.btn_obs_pause then
    app.btn_obs_pause:set_label(app.observer_paused and "▶ Resume" or "⏸ Pause")
  end
  if app.lbl_obs_status then
    if app.board.game_over then
      app.lbl_obs_status:set_text("Match finished")
    elseif app.observer_paused then
      app.lbl_obs_status:set_text("Paused (inspecting)")
    else
      app.lbl_obs_status:set_text(string.format("Observing Move %d...", app.board.current_ply + 1))
    end
  end

  local b_turn = (not app.board.game_over and app.board.turn == board.BLACK)
  local w_turn = (not app.board.game_over and app.board.turn == board.WHITE)
  app.lbl_black_turn:set_text(b_turn and "● TURN" or "")
  app.lbl_white_turn:set_text(w_turn and "● TURN" or "")

  app.lbl_opening:set_text("Opening: " .. app.board:get_opening_name())
  app.lbl_ply:set_text(string.format("Move %d / %d", app.board.current_ply, #app.board.history))

  -- Engine Stats
  local st = app.engine.stats
  local black_q = app.black_q or 0.0
  local wr = q_to_winrate(black_q)
  app.lbl_nodes:set_text(fmt_int(st.nodes))
  app.lbl_nps:set_text(fmt_int(st.nps) .. " NPS")
  app.lbl_time:set_text(tostring(st.time or 0) .. " ms")
  app.lbl_winrate:set_text(string.format("B %.1f%% / W %.1f%%", wr, 100 - wr))
  app.lbl_qval:set_text(string.format("%+.3f", black_q))
  local eval_str = tostring(st.evaluator or "uniform")
  if eval_str:find("onnx") then
    local arch = eval_str:match("onnx%-%S+%s+(%S+)") or "onnx"
    local steps = eval_str:match("training_steps%s+(%d+)")
    eval_str = steps and (arch .. " (" .. steps .. ")") or arch
  elseif eval_str:find("uniform") then
    eval_str = "uniform"
  end
  app.lbl_eval_name:set_text(eval_str)
  local show_ai_moves = app.can_show_ai_moves()
  app.lbl_pv:set_text((show_ai_moves and st.pv and st.pv ~= "") and st.pv:upper() or "(none)")

  if app.engine.state == "searching" then
    app.lbl_engine_pill:set_text("● THINKING")
  elseif app.engine.state == "ready" then
    app.lbl_engine_pill:set_text("● READY")
  else
    app.lbl_engine_pill:set_text("○ " .. tostring(app.engine.state):upper())
  end
  if app.btn_stop then
    app.btn_stop:set_sensitive(app.engine.state == "searching")
  end

  -- Update Notation TreeView
  app.moves_store:clear()
  local hist = app.board.history
  for i = 1, #hist, 2 do
    local m1 = hist[i]
    local m2 = hist[i + 1]
    local move_no = math.floor((i + 1) / 2)
    local b_str = (i == app.board.current_ply and "▸ " or "") .. m1.uci:upper()
    local w_str = m2 and ((i + 1 == app.board.current_ply and "▸ " or "") .. m2.uci:upper()) or ""
    local latest_q = m2 and (m2.eval_score or 0) or (m1.eval_score or 0)
    local adv_str = string.format("%.0f%% (%+.2f)", q_to_winrate(latest_q), latest_q)
    local target_ply = m2 and (i + 1) or i
    app.moves_store:append({ move_no, b_str, w_str, adv_str, target_ply })
  end

  -- Update Candidate Moves TreeView
  app.cand_store:clear()
  if show_ai_moves and st.candidates then
    for idx, c in ipairs(st.candidates) do
      app.cand_store:append({
        idx,
        c.move:upper(),
        string.format("%s (%.1f%%)", fmt_int(c.visits), c.pct or 0),
        string.format("%.1f%%", q_to_winrate(c.value)),
        string.format("%+.3f", c.value or 0),
        string.format("%.1f%%", c.prior or 0)
      })
    end
  end

  if app.btn_setup_turn then
    app.btn_setup_turn:set_label("Turn: " .. (app.board.turn == board.BLACK and "Black" or "White"))
  end

  if app.curve_frame then
    app.curve_frame:set_visible(app.show_graph)
  end

  app.board_da:queue_draw()
  app.gauge_da:queue_draw()
  app.graph_da:queue_draw()
end

-- ---------------------------------------------------------------------------
-- Build Pure GTK3 Native Window & Widgets
-- ---------------------------------------------------------------------------
local CSS = [[
window { background-color: #f6f8fa; color: #1f2328; }
menubar { background-color: #ffffff; border-bottom: 1px solid #e1e4e8; padding: 2px 6px; }
button, checkbutton { padding: 3px 8px; min-height: 24px; font-size: 11px; border-radius: 5px; }
combobox button { padding: 3px 8px; min-height: 24px; font-size: 11px; border-radius: 5px; }
.toolbar-card { background-color: #ffffff; border-bottom: 1px solid #e1e4e8; padding: 4px 10px; }
.card { background-color: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 8px 10px; }
.score-num { font-size: 24px; font-weight: 700; color: #1f2328; }
.turn-pill { font-size: 10px; font-weight: 700; color: #0969da; }
.metric-val { font-family: monospace; font-size: 13px; font-weight: 700; color: #1f2328; }
.metric-lbl { font-size: 10px; color: #59636e; }
.pv-box { background-color: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px; padding: 6px 8px; font-family: monospace; font-size: 11px; }
.statusbar { background-color: #ffffff; border-top: 1px solid #e1e4e8; padding: 3px 10px; font-size: 11px; color: #59636e; }
notebook tab { padding: 5px 12px; font-size: 11px; font-weight: 600; }
]]

local function add_menu_item(menu, label, accel_group, key, mods, cb)
  local item = Gtk.MenuItem.new_with_label(label)
  if accel_group and key then
    item:add_accelerator("activate", accel_group, Gdk.keyval_from_name(key), mods or 0, Gtk.AccelFlags.VISIBLE)
  end
  item.on_activate = cb
  menu:append(item)
  return item
end

local function make_col(title, col_idx, width)
  local renderer = Gtk.CellRendererText()
  local col = Gtk.TreeViewColumn { title = title, resizable = true }
  col:pack_start(renderer, true)
  col:add_attribute(renderer, "text", col_idx)
  if width then col:set_min_width(width) end
  return col
end

function app.build_ui()
  local settings = Gtk.Settings.get_default()
  settings.gtk_application_prefer_dark_theme = false

  local provider = Gtk.CssProvider()
  provider:load_from_data(CSS)
  Gtk.StyleContext.add_provider_for_screen(
    Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
  )

  local def_w, def_h = 1040, 680
  local disp = Gdk.Display.get_default()
  if disp then
    local mon = disp:get_primary_monitor() or disp:get_monitor(0)
    if mon then
      local wa = mon:get_workarea()
      if wa and wa.width > 0 and wa.height > 0 then
        def_w = math.min(1060, wa.width - 40)
        def_h = math.min(700, wa.height - 40)
      end
    end
  end

  local win = Gtk.Window {
    title = "Islay Othello",
    default_width = def_w,
    default_height = def_h,
    window_position = Gtk.WindowPosition.CENTER,
    on_destroy = function()
      app.engine:quit()
      Gtk.main_quit()
    end
  }
  app.win = win

  local accel = Gtk.AccelGroup()
  win:add_accel_group(accel)

  local root_vbox = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 0 }
  win:add(root_vbox)

  -- 1. Native Gtk.MenuBar
  local menubar = Gtk.MenuBar()
  local ctrl_mask = Gdk.ModifierType.CONTROL_MASK

  local file_menu = Gtk.Menu()
  add_menu_item(file_menu, "New Game", accel, "n", ctrl_mask, function() app.new_game() end)
  file_menu:append(Gtk.SeparatorMenuItem())
  add_menu_item(file_menu, "Copy FEN", accel, "c", ctrl_mask, function() app.copy_fen() end)
  add_menu_item(file_menu, "Paste FEN", accel, "v", ctrl_mask, function() app.paste_fen() end)
  file_menu:append(Gtk.SeparatorMenuItem())
  add_menu_item(file_menu, "Quit", accel, "q", ctrl_mask, function() win:destroy() end)
  local file_root = Gtk.MenuItem.new_with_label("File")
  file_root:set_submenu(file_menu)
  menubar:append(file_root)

  local game_menu = Gtk.Menu()
  add_menu_item(game_menu, "Play Black vs Islay", nil, nil, nil, function() app.set_mode(1) end)
  add_menu_item(game_menu, "Play White vs Islay", nil, nil, nil, function() app.set_mode(2) end)
  add_menu_item(game_menu, "Two Players (Manual)", nil, nil, nil, function() app.set_mode(3) end)
  add_menu_item(game_menu, "Self-Play (AI vs AI)", nil, nil, nil, function() app.set_mode(4) end)
  add_menu_item(game_menu, "Observer (Watch & Step)", nil, nil, nil, function() app.set_mode(5) end)
  game_menu:append(Gtk.SeparatorMenuItem())
  add_menu_item(game_menu, "Takeback (Undo)", accel, "z", ctrl_mask, function() app.undo() end)
  add_menu_item(game_menu, "Step Forward (Redo)", accel, "y", ctrl_mask, function() app.redo() end)
  add_menu_item(game_menu, "Pass Turn", accel, "space", 0, function() app.do_pass() end)
  local game_root = Gtk.MenuItem.new_with_label("Game")
  game_root:set_submenu(game_menu)
  menubar:append(game_root)

  local eng_menu = Gtk.Menu()
  add_menu_item(eng_menu, "Hint Best Move", accel, "h", 0, function() app.request_hint() end)
  add_menu_item(eng_menu, "Toggle Live MCTS (go infinite)", accel, "a", 0, function()
    app.btn_analyze:set_active(not app.btn_analyze:get_active())
  end)
  add_menu_item(eng_menu, "Stop Search", accel, "s", 0, function() app.stop_search() end)
  eng_menu:append(Gtk.SeparatorMenuItem())
  add_menu_item(eng_menu, "Run Perft (Depth 5)", accel, "p", 0, function() app.run_perft(5) end)
  eng_menu:append(Gtk.SeparatorMenuItem())
  add_menu_item(eng_menu, "Load Eval File...", accel, "o", ctrl_mask, function()
    local fn = app.load_eval_file_dialog(app.win)
    if fn then app.load_eval_file(fn) end
  end)
  add_menu_item(eng_menu, "Engine Settings...", accel, "comma", ctrl_mask, function() app.open_engine_settings() end)
  local eng_root = Gtk.MenuItem.new_with_label("Engine")
  eng_root:set_submenu(eng_menu)
  menubar:append(eng_root)

  local view_menu = Gtk.Menu()
  add_menu_item(view_menu, "Toggle Move Numbers", accel, "1", ctrl_mask, function()
    app.show_numbers = not app.show_numbers; app.board_da:queue_draw()
  end)
  add_menu_item(view_menu, "Toggle MCTS Overlay", accel, "2", ctrl_mask, function()
    app.show_mcts_overlay = not app.show_mcts_overlay; app.board_da:queue_draw()
  end)
  add_menu_item(view_menu, "Toggle Winrate Curve", accel, "3", ctrl_mask, function()
    app.show_graph = not app.show_graph; app.refresh_ui()
  end)
  local view_root = Gtk.MenuItem.new_with_label("View")
  view_root:set_submenu(view_menu)
  menubar:append(view_root)

  root_vbox:pack_start(menubar, false, false, 0)

  -- 2. Native Toolbar (Clean & Uncluttered)
  local tb = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 6 }
  tb:get_style_context():add_class("toolbar-card")

  local function mk_btn(lbl, cb)
    local b = Gtk.Button.new_with_label(lbl)
    b.on_clicked = cb
    tb:pack_start(b, false, false, 0)
    return b
  end

  mk_btn("New", function() app.new_game() end)
  mk_btn("|◀", function() app.goto_ply(0) end)
  mk_btn("◀", function() app.undo() end)
  mk_btn("▶", function() app.redo() end)
  mk_btn("▶|", function() app.goto_ply(#app.board.history) end)
  tb:pack_start(Gtk.Separator { orientation = Gtk.Orientation.VERTICAL }, false, false, 4)
  mk_btn("Pass", function() app.do_pass() end)
  mk_btn("Hint", function() app.request_hint() end)

  app.btn_analyze = Gtk.ToggleButton.new_with_label("Analyze (∞)")
  app.btn_analyze.on_toggled = function() app.toggle_analyze() end
  tb:pack_start(app.btn_analyze, false, false, 0)

  app.btn_stop = mk_btn("Stop", function() app.stop_search() end)
  app.btn_stop:set_sensitive(false)

  mk_btn("Perft", function() app.run_perft(5) end)
  mk_btn("Eval File", function()
    local fn = app.load_eval_file_dialog(app.win)
    if fn then app.load_eval_file(fn) end
  end)
  mk_btn("Settings", function() app.open_engine_settings() end)

  -- Right-side Native ComboBoxes (Mode, Rule, Budget)
  app.budget_combo = Gtk.ComboBoxText()
  for _, b in ipairs(app.budgets) do app.budget_combo:append_text(b.label) end
  app.budget_combo:set_active(app.selected_budget - 1)
  app.budget_combo.on_changed = function()
    if not app.suppress_combos then
      app.selected_budget = app.budget_combo:get_active() + 1
    end
  end
  tb:pack_end(app.budget_combo, false, false, 0)

  app.rule_combo = Gtk.ComboBoxText()
  app.rule_combo:append_text("Othello")
  app.rule_combo:append_text("Reversi")
  app.rule_combo:set_active(0)
  app.rule_combo.on_changed = function()
    if not app.suppress_combos then
      app.set_rule(app.rule_combo:get_active() == 1 and "Reversi" or "Othello")
    end
  end
  tb:pack_end(app.rule_combo, false, false, 0)

  app.mode_combo = Gtk.ComboBoxText()
  app.mode_combo:append_text("Black vs Islay")
  app.mode_combo:append_text("White vs Islay")
  app.mode_combo:append_text("2 Players")
  app.mode_combo:append_text("AI vs AI")
  app.mode_combo:append_text("Observer")
  app.mode_combo:set_active(0)
  app.mode_combo.on_changed = function()
    if not app.suppress_combos then
      app.set_mode(app.mode_combo:get_active() + 1)
    end
  end
  tb:pack_end(app.mode_combo, false, false, 0)

  root_vbox:pack_start(tb, false, false, 0)

  -- 3. Main Workspace (Left: Gauge + Board | Right: Scoreboard + Notebook + Winrate Graph)
  local main_hbox = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 12, margin = 10 }
  root_vbox:pack_start(main_hbox, true, true, 0)

  -- Left Stage
  local left_vbox = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 6 }
  main_hbox:pack_start(left_vbox, true, true, 0)

  local stage_hdr = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 8 }
  app.lbl_opening = Gtk.Label { label = "Opening: Standard Start Position", xalign = 0 }
  app.lbl_ply = Gtk.Label { label = "Move 0 / 0", xalign = 1 }
  stage_hdr:pack_start(app.lbl_opening, true, true, 0)
  stage_hdr:pack_end(app.lbl_ply, false, false, 0)
  left_vbox:pack_start(stage_hdr, false, false, 0)

  -- Horizontal Eval Bar above board
  app.gauge_da = Gtk.DrawingArea { height_request = 26 }
  app.gauge_da.on_draw = on_draw_gauge
  left_vbox:pack_start(app.gauge_da, false, false, 0)

  app.board_da = Gtk.DrawingArea { width_request = 420, height_request = 420 }
  app.board_da:add_events(
    Gdk.EventMask.BUTTON_PRESS_MASK +
    Gdk.EventMask.POINTER_MOTION_MASK +
    Gdk.EventMask.LEAVE_NOTIFY_MASK
  )
  app.board_da.on_draw = on_draw_board
  app.board_da.on_button_press_event = function(widget, ev)
    if ev.button == 1 then
      local alloc = widget:get_allocation()
      local ox, oy, _, margin, _, cs = board_geometry(alloc.width, alloc.height)
      local gx, gy = ox + margin, oy + margin
      local x = math.floor((ev.x - gx) / cs) + 1
      local y = math.floor((ev.y - gy) / cs) + 1
      if x >= 1 and x <= 8 and y >= 1 and y <= 8 then
        app.play_human_move(x, y)
      end
    end
    return true
  end
  app.board_da.on_motion_notify_event = function(widget, ev)
    local alloc = widget:get_allocation()
    local ox, oy, _, margin, _, cs = board_geometry(alloc.width, alloc.height)
    local gx, gy = ox + margin, oy + margin
    local x = math.floor((ev.x - gx) / cs) + 1
    local y = math.floor((ev.y - gy) / cs) + 1
    if x >= 1 and x <= 8 and y >= 1 and y <= 8 then
      if not app.hovered_cell or app.hovered_cell.x ~= x or app.hovered_cell.y ~= y then
        app.hovered_cell = { x = x, y = y }
        widget:queue_draw()
      end
    elseif app.hovered_cell then
      app.hovered_cell = nil
      widget:queue_draw()
    end
    return false
  end
  app.board_da.on_leave_notify_event = function(widget)
    app.hovered_cell = nil
    widget:queue_draw()
    return false
  end
  left_vbox:pack_start(app.board_da, true, true, 0)

  -- Clean Board Options strip below board
  local board_opts = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 10 }
  local chk_num = Gtk.CheckButton.new_with_label("Move Numbers")
  chk_num:set_active(true)
  chk_num.on_toggled = function() app.show_numbers = chk_num:get_active(); app.board_da:queue_draw() end
  board_opts:pack_start(chk_num, false, false, 0)

  local chk_mcts = Gtk.CheckButton.new_with_label("MCTS Winrate")
  chk_mcts:set_active(true)
  chk_mcts.on_toggled = function() app.show_mcts_overlay = chk_mcts:get_active(); app.board_da:queue_draw() end
  board_opts:pack_start(chk_mcts, false, false, 0)

  local chk_autoeval = Gtk.CheckButton.new_with_label("Auto Eval")
  chk_autoeval:set_active(true)
  chk_autoeval.on_toggled = function()
    app.auto_eval = chk_autoeval:get_active()
    if app.auto_eval and not app.is_ai_turn() then
      app.trigger_auto_eval()
    end
  end
  board_opts:pack_start(chk_autoeval, false, false, 0)
  left_vbox:pack_start(board_opts, false, false, 0)

  -- Observer Controls Strip (visible in Observer / AI vs AI modes)
  app.observer_bar = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 8 }
  app.observer_bar:get_style_context():add_class("card")
  app.btn_obs_pause = Gtk.Button.new_with_label("⏸ Pause")
  app.btn_obs_pause.on_clicked = function() app.observer_toggle_pause() end
  local btn_obs_step = Gtk.Button.new_with_label("⏭ Step")
  btn_obs_step.on_clicked = function() app.observer_step_once() end
  local pace_combo = Gtk.ComboBoxText()
  local paces = { 200, 600, 1200, 2000 }
  pace_combo:append_text("Pace: 0.2s")
  pace_combo:append_text("Pace: 0.6s")
  pace_combo:append_text("Pace: 1.2s")
  pace_combo:append_text("Pace: 2.0s")
  pace_combo:set_active(1)
  pace_combo.on_changed = function()
    local idx = pace_combo:get_active() + 1
    if paces[idx] then app.engine_opts.observer_pace_ms = paces[idx] end
  end
  app.lbl_obs_status = Gtk.Label { label = "Observing...", xalign = 1 }
  app.lbl_obs_status:get_style_context():add_class("metric-lbl")
  app.observer_bar:pack_start(app.btn_obs_pause, false, false, 0)
  app.observer_bar:pack_start(btn_obs_step, false, false, 0)
  app.observer_bar:pack_start(pace_combo, false, false, 0)
  app.observer_bar:pack_end(app.lbl_obs_status, true, true, 0)
  left_vbox:pack_start(app.observer_bar, false, false, 0)

  -- Right Workbench Stage
  local right_vbox = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 8, width_request = 360 }
  main_hbox:pack_start(right_vbox, false, true, 0)

  -- Scoreboard Card
  local score_card = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 12 }
  score_card:get_style_context():add_class("card")

  local function mk_player_box(name)
    local bx = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 2 }
    local top = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 6 }
    local lbl_n = Gtk.Label { label = name, xalign = 0 }
    local lbl_t = Gtk.Label { label = "", xalign = 0 }
    lbl_t:get_style_context():add_class("turn-pill")
    top:pack_start(lbl_n, false, false, 0)
    top:pack_start(lbl_t, false, false, 0)
    local lbl_s = Gtk.Label { label = "Human", xalign = 0 }
    lbl_s:get_style_context():add_class("metric-lbl")
    bx:pack_start(top, false, false, 0)
    bx:pack_start(lbl_s, false, false, 0)
    local sc = Gtk.Label { label = "2", xalign = 1 }
    sc:get_style_context():add_class("score-num")
    local row = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 8 }
    row:pack_start(bx, true, true, 0)
    row:pack_end(sc, false, false, 4)
    return row, sc, lbl_s, lbl_t
  end

  local b_row, b_sc, b_sub, b_trn = mk_player_box("● Black")
  local w_row, w_sc, w_sub, w_trn = mk_player_box("○ White")
  app.lbl_black_score, app.lbl_black_sub, app.lbl_black_turn = b_sc, b_sub, b_trn
  app.lbl_white_score, app.lbl_white_sub, app.lbl_white_turn = w_sc, w_sub, w_trn

  score_card:pack_start(b_row, true, true, 0)
  score_card:pack_start(Gtk.Separator { orientation = Gtk.Orientation.VERTICAL }, false, false, 2)
  score_card:pack_start(w_row, true, true, 0)
  right_vbox:pack_start(score_card, false, false, 0)

  -- Native Gtk.Notebook (Notation | Engine MCTS Tree | Position Setup)
  app.notebook = Gtk.Notebook()
  right_vbox:pack_start(app.notebook, true, true, 0)

  -- Tab 1: Notation Table
  app.moves_store = Gtk.ListStore.new {
    GObject.Type.INT, GObject.Type.STRING, GObject.Type.STRING, GObject.Type.STRING, GObject.Type.INT
  }
  app.moves_view = Gtk.TreeView { model = app.moves_store }
  app.moves_view:append_column(make_col("#", 0, 40))
  app.moves_view:append_column(make_col("Black", 1, 90))
  app.moves_view:append_column(make_col("White", 2, 90))
  app.moves_view:append_column(make_col("Win% (Q)", 3, 120))
  app.moves_view.on_row_activated = function(view, path)
    local iter = app.moves_store:get_iter(path)
    if iter then
      local ply = app.moves_store:get_value(iter, 4):get_int()
      app.goto_ply(ply)
    end
  end
  local moves_scroll = Gtk.ScrolledWindow { hscrollbar_policy = Gtk.PolicyType.NEVER }
  moves_scroll:add(app.moves_view)
  app.notebook:append_page(moves_scroll, Gtk.Label { label = "Notation" })

  -- Tab 2: AlphaGo MCTS Engine Inspector
  local eng_box = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 8, margin = 8 }
  local metrics_grid = Gtk.Grid { column_spacing = 12, row_spacing = 6, column_homogeneous = true }
  local function mk_metric(col, row, title)
    local vb = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 2 }
    vb:get_style_context():add_class("card")
    local l1 = Gtk.Label { label = title, xalign = 0 }
    l1:get_style_context():add_class("metric-lbl")
    local l2 = Gtk.Label { label = "0", xalign = 0, max_width_chars = 16, ellipsize = 3 }
    l2:get_style_context():add_class("metric-val")
    vb:pack_start(l1, false, false, 0)
    vb:pack_start(l2, false, false, 0)
    metrics_grid:attach(vb, col, row, 1, 1)
    return l2
  end
  app.lbl_nodes     = mk_metric(0, 0, "Simulations")
  app.lbl_nps       = mk_metric(1, 0, "Speed")
  app.lbl_time      = mk_metric(2, 0, "Elapsed")
  app.lbl_winrate   = mk_metric(0, 1, "Winrate")
  app.lbl_qval      = mk_metric(1, 1, "Q-Value (Black)")
  app.lbl_eval_name = mk_metric(2, 1, "Evaluator")
  eng_box:pack_start(metrics_grid, false, false, 0)

  eng_box:pack_start(Gtk.Label { label = "Principal Variation (PV):", xalign = 0 }, false, false, 0)
  app.lbl_pv = Gtk.Label { label = "(none)", xalign = 0, max_width_chars = 40, ellipsize = 3 }
  app.lbl_pv:get_style_context():add_class("pv-box")
  eng_box:pack_start(app.lbl_pv, false, false, 0)

  eng_box:pack_start(Gtk.Label { label = "MCTS Candidate Moves:", xalign = 0 }, false, false, 0)
  app.cand_store = Gtk.ListStore.new {
    GObject.Type.INT, GObject.Type.STRING, GObject.Type.STRING,
    GObject.Type.STRING, GObject.Type.STRING, GObject.Type.STRING
  }
  local cand_view = Gtk.TreeView { model = app.cand_store }
  cand_view:append_column(make_col("#", 0, 30))
  cand_view:append_column(make_col("Move", 1, 50))
  cand_view:append_column(make_col("Visits (%)", 2, 110))
  cand_view:append_column(make_col("Win%", 3, 60))
  cand_view:append_column(make_col("Q", 4, 65))
  cand_view:append_column(make_col("Prior", 5, 60))
  local cand_scroll = Gtk.ScrolledWindow { hscrollbar_policy = Gtk.PolicyType.NEVER }
  cand_scroll:add(cand_view)
  eng_box:pack_start(cand_scroll, true, true, 0)
  app.notebook:append_page(eng_box, Gtk.Label { label = "Engine MCTS" })

  -- Tab 3: Position Setup
  local setup_box = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 10, margin = 12 }
  local chk_setup = Gtk.CheckButton.new_with_label("Enable Board Edit Mode (Click squares to place)")
  chk_setup.on_toggled = function()
    app.board.setup_mode = chk_setup:get_active()
    app.set_status(app.board.setup_mode and "Position Setup Mode active" or "Resumed match")
    if not app.board.setup_mode then
      app.sync_engine()
      app.restart_analyze_if_active()
    end
    app.refresh_ui()
  end
  setup_box:pack_start(chk_setup, false, false, 0)

  local pal_row = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 6 }
  local r_b = Gtk.RadioButton.new_with_label(nil, "● Black")
  local r_w = Gtk.RadioButton.new_with_label_from_widget(r_b, "○ White")
  local r_e = Gtk.RadioButton.new_with_label_from_widget(r_b, "✕ Empty")
  r_b.on_toggled = function() if r_b:get_active() then app.setup_tool = board.BLACK end end
  r_w.on_toggled = function() if r_w:get_active() then app.setup_tool = board.WHITE end end
  r_e.on_toggled = function() if r_e:get_active() then app.setup_tool = board.EMPTY end end
  pal_row:pack_start(r_b, true, true, 0)
  pal_row:pack_start(r_w, true, true, 0)
  pal_row:pack_start(r_e, true, true, 0)
  setup_box:pack_start(pal_row, false, false, 0)

  local act_row = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 6 }
  app.btn_setup_turn = Gtk.Button.new_with_label("Turn: Black")
  app.btn_setup_turn.on_clicked = function()
    app.board.turn = app.board:opponent(app.board.turn)
    app.board:rebuild_rep_from_grid()
    app.refresh_ui()
  end
  local btn_inv = Gtk.Button.new_with_label("Invert")
  btn_inv.on_clicked = function() app.board:invert_board(); app.refresh_ui() end
  local btn_clr = Gtk.Button.new_with_label("Clear")
  btn_clr.on_clicked = function() app.board:clear_board(); app.refresh_ui() end
  local btn_apply = Gtk.Button.new_with_label("Apply & Resume")
  btn_apply.on_clicked = function()
    chk_setup:set_active(false)
    app.board.setup_mode = false
    app.sync_engine()
    app.restart_analyze_if_active()
    app.refresh_ui()
  end
  act_row:pack_start(app.btn_setup_turn, true, true, 0)
  act_row:pack_start(btn_inv, true, true, 0)
  act_row:pack_start(btn_clr, true, true, 0)
  setup_box:pack_start(act_row, false, false, 0)
  setup_box:pack_start(btn_apply, false, false, 0)
  app.notebook:append_page(setup_box, Gtk.Label { label = "Position Setup" })

  -- Winrate Trajectory Card
  app.curve_frame = Gtk.Box { orientation = Gtk.Orientation.VERTICAL, spacing = 4 }
  app.curve_frame:get_style_context():add_class("card")
  app.curve_frame:pack_start(Gtk.Label { label = "Winrate Trajectory (Click to jump)", xalign = 0 }, false, false, 0)
  app.graph_da = Gtk.DrawingArea { height_request = 96 }
  app.graph_da:add_events(
    Gdk.EventMask.BUTTON_PRESS_MASK +
    Gdk.EventMask.POINTER_MOTION_MASK +
    Gdk.EventMask.LEAVE_NOTIFY_MASK
  )
  app.graph_da.on_draw = on_draw_graph
  app.graph_da.on_motion_notify_event = function(widget, ev)
    if #app.board.history > 0 then
      local alloc = widget:get_allocation()
      local pad_x = 12
      local frac = (ev.x - pad_x) / math.max(1, alloc.width - pad_x * 2)
      local target = math.floor(frac * #app.board.history + 0.5)
      target = math.max(0, math.min(#app.board.history, target))
      if app.graph_hover_ply ~= target then
        app.graph_hover_ply = target
        widget:queue_draw()
      end
    end
    return false
  end
  app.graph_da.on_leave_notify_event = function(widget)
    if app.graph_hover_ply ~= nil then
      app.graph_hover_ply = nil
      widget:queue_draw()
    end
    return false
  end
  app.graph_da.on_button_press_event = function(widget, ev)
    if #app.board.history > 0 and ev.button == 1 then
      local alloc = widget:get_allocation()
      local pad_x = 12
      local frac = (ev.x - pad_x) / math.max(1, alloc.width - pad_x * 2)
      local target = math.floor(frac * #app.board.history + 0.5)
      target = math.max(0, math.min(#app.board.history, target))
      app.goto_ply(target)
    end
    return true
  end
  app.curve_frame:pack_start(app.graph_da, false, false, 0)
  right_vbox:pack_end(app.curve_frame, false, false, 0)

  -- 4. Native Statusbar
  local sb = Gtk.Box { orientation = Gtk.Orientation.HORIZONTAL, spacing = 8, margin = 4 }
  sb:get_style_context():add_class("statusbar")
  app.lbl_statusbar = Gtk.Label { label = app.status_text, xalign = 0 }
  app.lbl_engine_pill = Gtk.Label { label = "● INITIALIZING", xalign = 1 }
  sb:pack_start(app.lbl_statusbar, true, true, 0)
  sb:pack_end(app.lbl_engine_pill, false, false, 4)
  root_vbox:pack_end(sb, false, false, 0)

  win:show_all()
end

-- ---------------------------------------------------------------------------
-- Engine Callbacks & GLib Non-blocking Loop
-- ---------------------------------------------------------------------------
app.engine.on_ready = function()
  app.set_status("Engine ready: " .. (app.engine.binary_path or "islay"))
  if app.is_ai_turn() then
    app.ai_delay = 0.25
  elseif app.auto_eval then
    app.ai_delay = 0.15
  end
  app.refresh_ui()
end

app.engine.on_info = function(stats)
  if stats.value ~= nil then
    app.black_q = to_black_q(stats.value, app.searched_turn)
    if app.board.current_ply > 0 and app.board.history[app.board.current_ply] and app.searched_turn == app.board.turn then
      app.board.history[app.board.current_ply].eval_score = app.black_q
    end
  end
  app.refresh_ui()
end

app.engine.on_error = function(err)
  app.set_status("Engine error: " .. tostring(err))
  app.refresh_ui()
end

app.engine.on_perft_done = function(res)
  app.set_status(string.format("Perft(%d): %s nodes | %s | %s",
    res.depth, fmt_int(res.nodes), res.time_str, res.speed_str))
  app.refresh_ui()
end

app.engine.on_bestmove = function(bm, stats)
  if stats.value ~= nil then
    app.black_q = to_black_q(stats.value, app.searched_turn)
    if (app.analyzing or app.stop_only or app.eval_requested or app.hint_requested)
       and app.board.current_ply > 0 and app.board.history[app.board.current_ply]
       and app.searched_turn == app.board.turn then
      app.board.history[app.board.current_ply].eval_score = app.black_q
    end
  end

  if not bm or bm == "0000" then
    app.eval_requested = false
    app.hint_requested = false
    app.refresh_ui()
    return
  end

  if app.analyzing then
    app.refresh_ui()
    return
  end

  if app.stop_only then
    app.stop_only = false
    app.hint_move = bm
    app.set_status(string.format("Stopped: %s (Black Winrate %.1f%%, %s nodes)",
      bm:upper(), q_to_winrate(app.black_q), fmt_int(stats.nodes)))
    app.refresh_ui()
    return
  end

  if app.eval_requested then
    app.eval_requested = false
    app.hint_move = bm
    app.refresh_ui()
    return
  end

  if app.hint_requested then
    app.hint_requested = false
    app.hint_move = bm
    local wr = q_to_winrate(app.black_q)
    app.set_status(string.format("Hint: %s (Black Winrate %.1f%%, %s nodes)", bm:upper(), wr, fmt_int(stats.nodes)))
    app.refresh_ui()
    return
  end

  if bm == "pass" then
    app.board:pass(app.black_q)
    app.set_status("Islay passed turn")
  else
    local x, y = board.from_uci(bm)
    if x and y then
      local ok, err = app.board:apply_move(x, y, app.black_q)
      if ok then
        app.hint_move = nil
        app.set_status(string.format("Islay played %s (Black Winrate %.1f%%)",
          bm:upper(), q_to_winrate(app.black_q)))
      else
        app.set_status("Engine move error: " .. tostring(err))
      end
    end
  end

  if not app.board.game_over then
    if app.is_ai_turn() then
      app.ai_delay = (app.mode == 5) and ((app.engine_opts.observer_pace_ms or 600) / 1000.0) or 0.20
    elseif app.board.forced_pass then
      app.set_status("No legal moves — auto-passing...")
      app.ai_delay = 0.35
    elseif app.auto_eval and app.mode ~= 5 and app.mode ~= 4 then
      app.engine.stats.candidates = {}
      app.ai_delay = 0.08
    end
  end
  app.refresh_ui()
end

-- Start GUI & Engine
app.build_ui()
app.engine:start()
app.refresh_ui()

GLib.timeout_add(GLib.PRIORITY_DEFAULT, 16, function()
  app.engine:update()

  -- Smoothly animate Eval Bar toward target winrate during thinking/searching
  local target_wr = q_to_winrate(app.black_q or 0.0)
  local diff = target_wr - (app.displayed_wr or 50.0)
  if math.abs(diff) > 0.05 then
    app.displayed_wr = app.displayed_wr + diff * 0.25
    app.gauge_da:queue_draw()
  elseif app.displayed_wr ~= target_wr then
    app.displayed_wr = target_wr
    app.gauge_da:queue_draw()
  end

  if app.ai_delay > 0 then
    app.ai_delay = app.ai_delay - 0.016
    if app.ai_delay <= 0 then
      if app.engine.state == "ready" then
        app.ai_delay = 0
        if not app.board.game_over and not app.board.setup_mode then
          if app.is_ai_turn() then
            app.trigger_ai()
          elseif app.board.forced_pass then
            app.do_pass()
          elseif app.auto_eval then
            app.trigger_auto_eval()
          end
        end
      else
        -- Engine is still finishing/stopping previous search; retry next tick
        app.ai_delay = 0.05
      end
    end
  end
  return true
end)

Gtk.main()

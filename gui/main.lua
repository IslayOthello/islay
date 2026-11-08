-- Islay Othello / Reversi - WZebra Desktop GUI
local board    = require("gui.board")
local engine   = require("gui.engine")
local theme    = require("gui.theme")
local ui       = require("gui.ui")
local openings = require("gui.openings")

local app = {
  board = nil,
  engine = nil,

  -- Play Modes
  mode = 1, -- 1: Human (Black) vs AI (White)
            -- 2: Human (White) vs AI (Black)
            -- 3: Two Players (Manual)
            -- 4: AI vs AI (Self-Play)

  -- WZebra Display Toggles
  show_numbers = true,      -- Move numbers on discs (1, 2, 3...)
  show_eval_overlay = true, -- Evaluation scores on legal move squares
  show_graph = true,        -- Advantage curve graph
  show_move_table = true,   -- Move notation table

  -- Thinking Budgets
  budgets = {
    { label = "1K nodes",  nodes = 1000 },
    { label = "10K nodes", nodes = 10000 },
    { label = "50K nodes", nodes = 50000 },
    { label = "500 ms",    movetime = 500 }
  },
  selected_budget = 2,

  -- State
  ai_delay = 0,
  hint_move = nil,
  hint_requested = false,
  evaluating_all = false,
  status_banner = nil,
  status_timer = 0,
  hovered_cell = nil,
  setup_tool = 1, -- 1: Black, 2: White, 0: Empty

  -- Modals & Popups
  show_perft = false,
  perft_depth = 5,
  show_about = false
}

function love.load()
  love.window.setTitle("Islay - Othello Engine (WZebra Edition)")
  love.window.setMode(theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT, { resizable = false, vsync = 1 })

  app.board = board.new("Othello")
  app.engine = engine.new()

  app.engine.on_ready = function()
    app.set_status("Engine ready: " .. (app.engine.binary_path or "islay"))
    if app.mode == 2 or app.mode == 4 then
      app.ai_delay = 0.3
    end
  end

  app.engine.on_bestmove = function(bm, stats)
    if not bm or bm == "0000" then return end

    if app.hint_requested then
      app.hint_move = bm
      app.hint_requested = false
      app.set_status("Islay suggests: " .. bm:upper())
      return
    end

    if app.evaluating_all then
      app.evaluating_all = false
      return
    end

    -- Record move evaluation for WZebra advantage graph
    local current_eval = stats.value or 0.0
    if app.board.turn == board.WHITE then
      current_eval = -current_eval
    end

    if bm == "pass" then
      app.board:pass(current_eval)
      app.set_status("Islay passed turn")
    else
      local x, y = board.from_uci(bm)
      if x and y then
        local ok, err = app.board:apply_move(x, y, current_eval)
        if ok then
          app.hint_move = nil
          app.set_status("Islay played " .. bm:upper())
        else
          app.set_status("Move error: " .. tostring(err))
        end
      end
    end
  end

  app.engine.on_error = function(err)
    app.set_status("Error: " .. tostring(err), 4.0)
  end

  app.engine.on_perft_done = function(results)
    app.set_status(string.format("Perft depth %d: %s nodes in %s (%s)",
      results.depth, tostring(results.nodes), results.time_str, results.speed_str), 5.0)
  end

  app.engine:start()
end

function app.set_status(msg, duration)
  app.status_banner = msg
  app.status_timer = duration or 3.5
end

function app.is_ai_turn()
  if app.board.game_over or app.board.setup_mode then return false end
  local turn = app.board.turn
  if app.mode == 1 and turn == board.WHITE then return true end
  if app.mode == 2 and turn == board.BLACK then return true end
  if app.mode == 4 then return true end
  return false
end

function love.update(dt)
  app.board:update_animations(dt)
  app.engine:update()

  if app.status_timer > 0 then
    app.status_timer = app.status_timer - dt
    if app.status_timer <= 0 then
      app.status_banner = nil
    end
  end

  local mx, my = love.mouse.getPosition()
  ui.update_input(mx, my, love.mouse.isDown(1), false)

  -- Track hovered board cell
  app.hovered_cell = nil
  if not app.show_perft and not app.show_about then
    local bx, by = theme.BOARD_X, theme.BOARD_Y
    local cs = theme.CELL_SIZE
    if mx >= bx and mx < bx + theme.BOARD_SIZE and my >= by and my < by + theme.BOARD_SIZE then
      local cx = math.floor((mx - bx) / cs) + 1
      local cy = math.floor((my - by) / cs) + 1

      if app.board.setup_mode then
        app.hovered_cell = { x = cx, y = cy, flips = {} }
      elseif not app.board.game_over and not app.is_ai_turn() then
        local legal_moves = app.board:get_legal_moves(app.board.turn)
        local uci = board.to_uci(cx, cy)
        if legal_moves[uci] then
          app.hovered_cell = { x = cx, y = cy, flips = legal_moves[uci].flips }
        end
      end
    end
  end

  -- AI move scheduling
  if app.is_ai_turn() and not app.board:is_animating() and app.engine.state == "ready" then
    if app.ai_delay > 0 then
      app.ai_delay = app.ai_delay - dt
    else
      app.trigger_ai_search()
    end
  end

  -- Forced pass handling
  if app.board.forced_pass and not app.board:is_animating() then
    if app.is_ai_turn() or app.mode == 4 then
      app.board:pass()
      app.set_status("Pass turn")
      app.ai_delay = 0.3
    end
  end
end

function app.trigger_ai_search()
  app.engine:sync_position(app.board:to_uci_moves())
  local b = app.budgets[app.selected_budget]
  if b.movetime then
    app.engine:search_movetime(b.movetime)
  else
    app.engine:search_nodes(b.nodes)
  end
end

function love.draw()
  local c = theme.colors
  ui.set_color(c.desktop_bg)
  love.graphics.rectangle("fill", 0, 0, theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT)

  -- 1. Desktop Menu Bar
  app.draw_menus()

  -- 2. Desktop Toolbar
  app.draw_toolbar()

  -- 3. Opening Name & Header
  app.draw_header()

  -- 4. Main WZebra Board
  app.draw_board()

  -- 5. Right Sidebar (Workspace Panels)
  app.draw_sidebar()

  -- 6. Status Bar at Bottom
  app.draw_statusbar()

  -- 7. Modals / Overlays
  if app.show_perft then
    app.draw_perft_modal()
  end

  if app.show_about then
    app.draw_about_modal()
  end
end

function app.draw_menus()
  local menus = {
    {
      title = "File",
      items = {
        { label = "New Game", shortcut = "Cmd+N", action = "new_game" },
        { label = "Copy Position FEN", shortcut = "Cmd+C", action = "copy_fen" },
        { label = "Paste Position FEN", shortcut = "Cmd+V", action = "paste_fen" },
        { separator = true },
        { label = "Exit", shortcut = "Cmd+Q", action = "quit" }
      }
    },
    {
      title = "Edit",
      items = {
        { label = "Setup Board Mode", checked = app.board.setup_mode, action = "toggle_setup" },
        { label = "Invert Board (Swap Colors)", action = "invert_board" },
        { label = "Clear Board", action = "clear_board" },
        { label = "Standard Startpos", action = "startpos" }
      }
    },
    {
      title = "Game",
      items = {
        { label = "Play as Black", checked = (app.mode == 1), action = "mode_1" },
        { label = "Play as White", checked = (app.mode == 2), action = "mode_2" },
        { label = "Two Players (Manual)", checked = (app.mode == 3), action = "mode_3" },
        { label = "Self-Play (AI vs AI)", checked = (app.mode == 4), action = "mode_4" },
        { separator = true },
        { label = "Takeback Move", shortcut = "Cmd+Z", disabled = not app.board:can_undo(), action = "undo" },
        { label = "Step Forward", shortcut = "Cmd+Y", disabled = not app.board:can_redo(), action = "redo" },
        { label = "Pass Turn", shortcut = "Space", disabled = not app.board.forced_pass, action = "pass" }
      }
    },
    {
      title = "Analysis",
      items = {
        { label = "Hint Best Move", shortcut = "H", action = "hint" },
        { label = "Analyze Position", shortcut = "A", action = "analyze" },
        { separator = true },
        { label = "Perft Benchmark", shortcut = "P", action = "perft" }
      }
    },
    {
      title = "View",
      items = {
        { label = "Move Numbers on Discs", checked = app.show_numbers, action = "toggle_numbers" },
        { label = "Evaluation on Legal Moves", checked = app.show_eval_overlay, action = "toggle_eval_overlay" },
        { label = "Advantage Curve Graph", checked = app.show_graph, action = "toggle_graph" },
        { label = "Move History Sheet", checked = app.show_move_table, action = "toggle_move_table" }
      }
    },
    {
      title = "Options",
      items = {
        { label = "Rule: Othello", checked = (app.board.rule == "Othello"), action = "rule_othello" },
        { label = "Rule: Reversi", checked = (app.board.rule == "Reversi"), action = "rule_reversi" },
        { separator = true },
        { label = "AI Level: Casual (1K nodes)", checked = (app.selected_budget == 1), action = "budget_1" },
        { label = "AI Level: Club (10K nodes)", checked = (app.selected_budget == 2), action = "budget_2" },
        { label = "AI Level: Master (50K nodes)", checked = (app.selected_budget == 3), action = "budget_3" },
        { label = "AI Level: Time (500 ms)", checked = (app.selected_budget == 4), action = "budget_4" }
      }
    },
    {
      title = "Help",
      items = {
        { label = "Keyboard Shortcuts", action = "shortcuts" },
        { label = "About Islay Othello", action = "about" }
      }
    }
  }

  local action = ui.draw_menubar(menus)
  if action then
    app.handle_action(action)
  end
end

function app.draw_toolbar()
  local c = theme.colors
  local w = theme.WINDOW_WIDTH
  local h = theme.TOOLBAR_H
  local y = theme.MENUBAR_H

  ui.draw_rect(0, y, w, h, c.toolbar_bg, c.divider)

  local cur_x = 10
  local btn_h = 26
  local by = y + 6

  -- File actions
  if ui.toolbar_button("tb_new", "New", cur_x, by, 50, btn_h) then
    app.handle_action("new_game")
  end
  cur_x = cur_x + 56

  if ui.toolbar_button("tb_undo", "Undo", cur_x, by, 52, btn_h, { disabled = not app.board:can_undo() }) then
    app.handle_action("undo")
  end
  cur_x = cur_x + 58

  if ui.toolbar_button("tb_redo", "Redo", cur_x, by, 52, btn_h, { disabled = not app.board:can_redo() }) then
    app.handle_action("redo")
  end
  cur_x = cur_x + 64

  -- VCR Navigation controls (WZebra iconic navigation buttons)
  if ui.toolbar_button("tb_start", "|<<", cur_x, by, 36, btn_h, { disabled = (app.board.current_ply == 0) }) then
    app.board:goto_start()
    app.engine:sync_position(app.board:to_uci_moves())
  end
  cur_x = cur_x + 40

  if ui.toolbar_button("tb_prev", "<", cur_x, by, 32, btn_h, { disabled = not app.board:can_undo() }) then
    app.handle_action("undo")
  end
  cur_x = cur_x + 36

  if ui.toolbar_button("tb_next", ">", cur_x, by, 32, btn_h, { disabled = not app.board:can_redo() }) then
    app.handle_action("redo")
  end
  cur_x = cur_x + 36

  if ui.toolbar_button("tb_end", ">>|", cur_x, by, 36, btn_h, { disabled = (app.board.current_ply == #app.board.history) }) then
    app.board:goto_end()
    app.engine:sync_position(app.board:to_uci_moves())
  end
  cur_x = cur_x + 48

  -- Setup mode toggle
  if ui.toolbar_button("tb_setup", "Setup Board", cur_x, by, 90, btn_h, { active = app.board.setup_mode }) then
    app.handle_action("toggle_setup")
  end
  cur_x = cur_x + 96

  -- WZebra view toggles
  if ui.toolbar_button("tb_num", "Numbers", cur_x, by, 72, btn_h, { active = app.show_numbers }) then
    app.show_numbers = not app.show_numbers
  end
  cur_x = cur_x + 78

  if ui.toolbar_button("tb_eval", "Eval Overlay", cur_x, by, 94, btn_h, { active = app.show_eval_overlay }) then
    app.show_eval_overlay = not app.show_eval_overlay
  end
  cur_x = cur_x + 100

  if ui.toolbar_button("tb_graph", "Graph", cur_x, by, 58, btn_h, { active = app.show_graph }) then
    app.show_graph = not app.show_graph
  end
  cur_x = cur_x + 68

  -- Analysis actions
  if ui.toolbar_button("tb_hint", "Hint", cur_x, by, 52, btn_h, { disabled = app.board.game_over }) then
    app.handle_action("hint")
  end
  cur_x = cur_x + 58

  if ui.toolbar_button("tb_bench", "Perft", cur_x, by, 56, btn_h) then
    app.show_perft = true
  end
end

function app.draw_header()
  local c = theme.colors
  local bx = theme.BOARD_X
  local by = theme.BOARD_Y - 24

  ui.set_color(c.text_secondary)
  love.graphics.print("Opening: ", bx, by)

  local op_name = app.board:get_opening_name()
  ui.set_color(c.accent_gold)
  love.graphics.print(op_name, bx + 64, by)

  local ply_info = string.format("Ply: %d / %d", app.board.current_ply, #app.board.history)
  ui.set_color(c.text_muted)
  love.graphics.print(ply_info, bx + theme.BOARD_SIZE - 90, by)
end

function app.draw_board()
  local c = theme.colors
  local bx, by = theme.BOARD_X, theme.BOARD_Y
  local bs = theme.BOARD_SIZE
  local cs = theme.CELL_SIZE
  local rim = theme.BORDER_THICK

  -- Outer border rim
  ui.draw_rect(bx - rim, by - rim, bs + rim * 2, bs + rim * 2, c.board_rim, c.board_rim_outer)

  -- Coordinates
  ui.set_color(c.coord_text)
  for i = 1, 8 do
    local ch = string.char(64 + i) -- A - H
    local cx = bx + (i - 0.5) * cs - 4
    love.graphics.print(ch, cx, by - rim + 2)
    love.graphics.print(ch, cx, by + bs + 2)

    local num_str = tostring(i)
    local cy = by + (i - 0.5) * cs - 6
    love.graphics.print(num_str, bx - rim + 5, cy)
    love.graphics.print(num_str, bx + bs + 5, cy)
  end

  -- Felt Interior
  ui.draw_rect(bx, by, bs, bs, c.board_felt, c.board_grid)

  -- Grid lines
  ui.set_color(c.board_grid)
  for i = 0, 8 do
    love.graphics.line(bx + i * cs, by, bx + i * cs, by + bs)
    love.graphics.line(bx, by + i * cs, bx + bs, by + i * cs)
  end

  -- Star points at (2,2), (6,2), (2,6), (6,6)
  local star_points = { {2, 2}, {6, 2}, {2, 6}, {6, 6} }
  for _, sp in ipairs(star_points) do
    ui.draw_circle(bx + sp[1] * cs, by + sp[2] * cs, 4, c.star_point)
  end

  -- Map of animating flip scales
  local anim_scales = {}
  for _, a in ipairs(app.board.animating_flips) do
    if not a.done and a.delay <= 0 then
      local progress = a.progress or 0
      local key = a.x .. "_" .. a.y
      local sx = math.abs(math.cos(progress * math.pi))
      local col = (progress < 0.5) and a.from_color or a.to_color
      anim_scales[key] = { scale_x = sx, color = col }
    end
  end

  -- Preview flipped discs
  local preview_flips = {}
  if app.hovered_cell and not app.is_ai_turn() and not app.board.setup_mode then
    for _, f in ipairs(app.hovered_cell.flips) do
      preview_flips[f.x .. "_" .. f.y] = true
    end
  end

  -- Draw Discs
  for y = 1, 8 do
    for x = 1, 8 do
      local val = app.board:get(x, y)
      local move_num = app.board:get_move_number(x, y)
      local cell_cx = bx + (x - 0.5) * cs
      local cell_cy = by + (y - 0.5) * cs
      local r = cs * 0.42

      local anim_info = anim_scales[x .. "_" .. y]
      if anim_info then
        ui.draw_disc(cell_cx, cell_cy, r, anim_info.color, move_num, app.show_numbers, anim_info.scale_x)
      elseif val ~= board.EMPTY then
        ui.draw_disc(cell_cx, cell_cy, r, val, move_num, app.show_numbers, 1.0)
        if preview_flips[x .. "_" .. y] then
          ui.draw_circle(cell_cx, cell_cy, r * 0.5, { 1.0, 0.8, 0.2, 0.55 })
        end
      end
    end
  end

  -- Last move highlight (warm gold ring)
  if app.board.last_move and app.board.last_move.x ~= board.PASS then
    local lx = bx + (app.board.last_move.x - 0.5) * cs
    local ly = by + (app.board.last_move.y - 0.5) * cs
    ui.draw_circle(lx, ly, cs * 0.44, nil, c.last_move)
    ui.draw_circle(lx, ly, 4, c.last_move)
  end

  -- Best move recommendation (cyan ring)
  if app.hint_move and app.hint_move ~= "pass" and app.hint_move ~= "0000" then
    local hx, hy = board.from_uci(app.hint_move)
    if hx and hy then
      local hcx = bx + (hx - 0.5) * cs
      local hcy = by + (hy - 0.5) * cs
      ui.draw_circle(hcx, hcy, cs * 0.44, nil, c.best_move_hint)
    end
  end

  -- Legal Move Hints & WZebra Evaluation Overlays
  if not app.board.game_over and not app.is_ai_turn() and not app.board.setup_mode then
    local legal_moves = app.board:get_legal_moves(app.board.turn)
    for _, m in pairs(legal_moves) do
      local dot_cx = bx + (m.x - 0.5) * cs
      local dot_cy = by + (m.y - 0.5) * cs

      if app.hovered_cell and app.hovered_cell.x == m.x and app.hovered_cell.y == m.y then
        ui.draw_circle(dot_cx, dot_cy, cs * 0.38, nil, c.legal_hint_hover)
        local count_txt = "+" .. tostring(#m.flips)
        ui.set_color(c.legal_hint_hover)
        love.graphics.print(count_txt, dot_cx - 8, dot_cy - 7)
      else
        ui.draw_circle(dot_cx, dot_cy, 6, c.legal_hint)
      end

      -- WZebra feature: Show evaluation score directly on legal move square!
      if app.show_eval_overlay and app.board.legal_evals[m.uci] then
        local score = app.board.legal_evals[m.uci]
        local score_str = string.format("%+.1f", score)
        local col = (score >= 0) and c.eval_square_pos or c.eval_square_neg
        ui.set_color(col)
        love.graphics.print(score_str, dot_cx - 14, dot_cy + 10)
      end
    end
  end
end

function app.draw_sidebar()
  local c = theme.colors
  local sx = theme.SIDEBAR_X
  local sy = theme.SIDEBAR_Y
  local sw = theme.SIDEBAR_W

  -- 1. Score & Turn Status Card
  local b_count, w_count = app.board:get_counts()
  local is_b_turn = (app.board.turn == board.BLACK and not app.board.game_over)
  local is_w_turn = (app.board.turn == board.WHITE and not app.board.game_over)

  ui.draw_rect(sx, sy, sw, 64, c.panel_bg, c.panel_border)

  -- Black side
  ui.draw_disc(sx + 24, sy + 32, 18, 1, nil, false)
  ui.set_color(c.text_primary)
  love.graphics.print(string.format("Black: %2d", b_count), sx + 50, sy + 16)
  if is_b_turn then
    ui.set_color(c.accent_gold)
    love.graphics.print("● To Move", sx + 50, sy + 36)
  end

  -- White side
  ui.draw_disc(sx + 180, sy + 32, 18, 2, nil, false)
  ui.set_color(c.text_primary)
  love.graphics.print(string.format("White: %2d", w_count), sx + 206, sy + 16)
  if is_w_turn then
    ui.set_color(c.accent_blue)
    love.graphics.print("● To Move", sx + 206, sy + 36)
  end

  -- Mode Badge
  local mode_names = { "vs AI (Black)", "vs AI (White)", "2-Player", "Self-Play" }
  ui.set_color(c.text_muted)
  love.graphics.print(mode_names[app.mode] or "", sx + sw - 110, sy + 24)

  local cur_y = sy + 72

  -- Setup Mode Palette (if active)
  if app.board.setup_mode then
    local new_tool, new_turn, action = ui.draw_setup_palette(sx, cur_y, sw, 172, app.setup_tool, app.board.turn)
    app.setup_tool = new_tool
    app.board.turn = new_turn

    if action == "clear_all" then
      app.board:clear_board()
    elseif action == "startpos" then
      app.board:reset()
      app.board.setup_mode = true
    elseif action == "invert" then
      app.board:invert_board()
    elseif action == "done" then
      app.board.setup_mode = false
      app.engine:sync_position(app.board:to_uci_moves())
      app.set_status("Position setup finished")
    end
    cur_y = cur_y + 180
  end

  -- 2. WZebra Advantage Curve Graph
  if app.show_graph then
    local graph_h = 138
    local clicked_ply = ui.draw_eval_graph(sx, cur_y, sw, graph_h, app.board.history, app.board.current_ply)
    if clicked_ply then
      app.board:goto_ply(clicked_ply)
      app.engine:sync_position(app.board:to_uci_moves())
    end
    cur_y = cur_y + graph_h + 8
  end

  -- 3. Move Notation Table
  if app.show_move_table then
    local table_h = app.show_graph and 170 or 310
    local clicked_ply = ui.draw_move_table(sx, cur_y, sw, table_h, app.board.history, app.board.current_ply)
    if clicked_ply then
      app.board:goto_ply(clicked_ply)
      app.engine:sync_position(app.board:to_uci_moves())
    end
    cur_y = cur_y + table_h + 8
  end

  -- 4. Engine Live Analysis Box
  local an_h = 76
  ui.draw_rect(sx, cur_y, sw, an_h, c.panel_bg, c.panel_border)

  ui.set_color(c.accent_blue)
  love.graphics.print("ENGINE ANALYSIS", sx + 10, cur_y + 8)

  local s = app.engine.stats
  ui.set_color(c.text_secondary)
  love.graphics.print(string.format("Nodes: %s  |  Speed: %s NPS  |  Time: %d ms",
    tostring(s.nodes), tostring(s.nps), s.time), sx + 10, cur_y + 26)

  local eval_display = string.format("Eval Score: %+.2f  |  Evaluations: %d", s.value, s.evaluations)
  ui.set_color((s.value >= 0) and c.eval_square_pos or c.eval_square_neg)
  love.graphics.print(eval_display, sx + 10, cur_y + 42)

  local pv_str = (s.pv and s.pv ~= "") and s.pv:upper() or "(none)"
  if #pv_str > 48 then pv_str = pv_str:sub(1, 48) .. "..." end
  ui.set_color(c.text_muted)
  love.graphics.print("Best Line: " .. pv_str, sx + 10, cur_y + 58)
end

function app.draw_statusbar()
  local pos_str = ""
  if app.hovered_cell then
    local col_char = string.char(64 + app.hovered_cell.x)
    pos_str = string.format("Square: %s%d", col_char, app.hovered_cell.y)
  end

  local center_msg = app.status_banner or ("Opening: " .. app.board:get_opening_name())
  local right_msg = "Engine: " .. app.engine.state:upper()
  ui.draw_statusbar(pos_str, center_msg, right_msg)
end

function app.draw_perft_modal()
  local c = theme.colors
  ui.draw_rect(0, 0, theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT, c.modal_bg)

  local mw, mh = 560, 420
  local mx = (theme.WINDOW_WIDTH - mw) / 2
  local my = (theme.WINDOW_HEIGHT - mh) / 2
  ui.draw_rect(mx, my, mw, mh, c.modal_card, c.accent_gold)

  ui.set_color(c.text_primary)
  love.graphics.print("Islay Perft Move-Generation Benchmark", mx + 20, my + 16)

  if ui.toolbar_button("p_close", "X", mx + mw - 36, my + 12, 24, 24) then
    app.show_perft = false
  end

  ui.set_color(c.text_secondary)
  love.graphics.print("Select Depth:", mx + 20, my + 54)
  for d = 1, 8 do
    local bx = mx + 130 + (d - 1) * 44
    if ui.toolbar_button("d_" .. d, tostring(d), bx, my + 48, 36, 26, { active = (app.perft_depth == d) }) then
      app.perft_depth = d
    end
  end

  if ui.toolbar_button("p_run", "Run Perft", mx + 20, my + 88, 120, 30, { active = true }) then
    app.engine:sync_position(app.board:to_uci_moves())
    app.engine:perft(app.perft_depth)
  end

  local res = app.engine.perft_results
  local ty = my + 130
  ui.draw_rect(mx + 20, ty, mw - 40, 220, c.panel_bg, c.panel_border)

  ui.set_color(c.text_secondary)
  love.graphics.print(string.format("Branches (Depth %d):", res.depth), mx + 30, ty + 10)

  local by_tab = ty + 32
  for i, b in ipairs(res.branches) do
    if i <= 8 then
      local line = string.format("%s: %s", b.move, tostring(b.count))
      ui.set_color(c.text_primary)
      love.graphics.print(line, mx + 30, by_tab)
      by_tab = by_tab + 18
    end
  end

  local summary_y = ty + 184
  ui.set_color(c.accent_gold)
  love.graphics.print("Total Nodes: " .. tostring(res.nodes), mx + 30, summary_y)
  love.graphics.print("Time: " .. (res.time_str or "-"), mx + 240, summary_y)
  love.graphics.print("Speed: " .. (res.speed_str or "-"), mx + 380, summary_y)
end

function app.draw_about_modal()
  local c = theme.colors
  ui.draw_rect(0, 0, theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT, c.modal_bg)

  local mw, mh = 480, 280
  local mx = (theme.WINDOW_WIDTH - mw) / 2
  local my = (theme.WINDOW_HEIGHT - mh) / 2
  ui.draw_rect(mx, my, mw, mh, c.modal_card, c.accent_gold)

  ui.set_color(c.accent_gold)
  love.graphics.print("About Islay Othello / Reversi Engine", mx + 24, my + 20)

  ui.set_color(c.text_primary)
  love.graphics.print("WZebra-Style Desktop Interface for islay 0.1.0", mx + 24, my + 54)

  ui.set_color(c.text_secondary)
  love.graphics.print("Features:", mx + 24, my + 84)
  love.graphics.print("- Othello & Reversi bitboard movegen (NEON / AVX2)", mx + 34, my + 104)
  love.graphics.print("- PUCT MCTS search with real-time UCI communication", mx + 34, my + 124)
  love.graphics.print("- WZebra advantage graph, move numbers & opening books", mx + 34, my + 144)
  love.graphics.print("- Built with Lua & LÖVE 2D", mx + 34, my + 164)

  if ui.toolbar_button("abt_close", "Close", mx + (mw - 100) / 2, my + 220, 100, 32, { active = true }) then
    app.show_about = false
  end
end

function app.handle_action(act)
  if act == "new_game" then
    app.board:reset()
    app.hint_move = nil
    app.engine:sync_position("")
    app.set_status("New game started")
    if app.mode == 2 or app.mode == 4 then app.ai_delay = 0.3 end
  elseif act == "copy_fen" then
    local fen, stm = app.board:to_fen()
    love.system.setClipboardText(fen .. " " .. stm)
    app.set_status("Position FEN copied to clipboard")
  elseif act == "paste_fen" then
    local text = love.system.getClipboardText()
    if text and #text >= 64 then
      local fen = text:sub(1, 64)
      local stm = text:match("%s+([XO])") or "X"
      local ok, err = app.board:load_fen(fen, stm)
      if ok then
        app.engine:sync_position(app.board:to_uci_moves(), fen, stm)
        app.set_status("Loaded FEN position")
      else
        app.set_status("FEN error: " .. tostring(err))
      end
    end
  elseif act == "quit" then
    love.event.quit()
  elseif act == "toggle_setup" then
    app.board.setup_mode = not app.board.setup_mode
    if not app.board.setup_mode then
      app.engine:sync_position(app.board:to_uci_moves())
    end
  elseif act == "invert_board" then
    app.board:invert_board()
    app.engine:sync_position(app.board:to_uci_moves())
    app.set_status("Board colors inverted")
  elseif act == "clear_board" then
    app.board:clear_board()
    app.set_status("Board cleared")
  elseif act == "startpos" then
    app.board:reset()
    app.engine:sync_position("")
    app.set_status("Reset to start position")
  elseif act == "undo" then
    app.board:undo()
    app.engine:sync_position(app.board:to_uci_moves())
  elseif act == "redo" then
    app.board:redo()
    app.engine:sync_position(app.board:to_uci_moves())
  elseif act == "pass" then
    app.board:pass()
    app.engine:sync_position(app.board:to_uci_moves())
  elseif act == "hint" then
    app.hint_requested = true
    app.engine:sync_position(app.board:to_uci_moves())
    app.engine:search_nodes(5000)
    app.set_status("Analyzing position for best move...")
  elseif act == "analyze" then
    app.evaluating_all = true
    app.engine:sync_position(app.board:to_uci_moves())
    app.engine:search_nodes(10000)
    app.set_status("Evaluating all candidates...")
  elseif act == "perft" then
    app.show_perft = true
  elseif act == "toggle_numbers" then
    app.show_numbers = not app.show_numbers
  elseif act == "toggle_eval_overlay" then
    app.show_eval_overlay = not app.show_eval_overlay
  elseif act == "toggle_graph" then
    app.show_graph = not app.show_graph
  elseif act == "toggle_move_table" then
    app.show_move_table = not app.show_move_table
  elseif act == "mode_1" then
    app.mode = 1; app.handle_action("new_game")
  elseif act == "mode_2" then
    app.mode = 2; app.handle_action("new_game")
  elseif act == "mode_3" then
    app.mode = 3; app.handle_action("new_game")
  elseif act == "mode_4" then
    app.mode = 4; app.handle_action("new_game")
  elseif act == "rule_othello" then
    app.board.rule = "Othello"; app.engine:set_rule("Othello"); app.set_status("Rule: Othello")
  elseif act == "rule_reversi" then
    app.board.rule = "Reversi"; app.engine:set_rule("Reversi"); app.set_status("Rule: Reversi")
  elseif act == "budget_1" then app.selected_budget = 1; app.set_status("AI Level: 1K nodes")
  elseif act == "budget_2" then app.selected_budget = 2; app.set_status("AI Level: 10K nodes")
  elseif act == "budget_3" then app.selected_budget = 3; app.set_status("AI Level: 50K nodes")
  elseif act == "budget_4" then app.selected_budget = 4; app.set_status("AI Level: 500 ms")
  elseif act == "shortcuts" then
    app.set_status("Cmd+N: New, Cmd+Z: Undo, Cmd+Y: Redo, H: Hint, P: Perft, Space: Pass", 6.0)
  elseif act == "about" then
    app.show_about = true
  end
end

function love.mousepressed(x, y, button)
  if button ~= 1 then return end
  ui.update_input(x, y, true, true)

  if ui.active_menu then return end
  if app.show_perft or app.show_about then return end

  -- Board interaction
  local bx, by = theme.BOARD_X, theme.BOARD_Y
  local bs = theme.BOARD_SIZE
  local cs = theme.CELL_SIZE

  if x >= bx and x < bx + bs and y >= by and y < by + bs then
    local cx = math.floor((x - bx) / cs) + 1
    local cy = math.floor((y - by) / cs) + 1

    if app.board.setup_mode then
      -- Position editor: directly set piece
      app.board:set_piece(cx, cy, app.setup_tool)
    elseif not app.board.game_over and not app.is_ai_turn() then
      local ok, err = app.board:apply_move(cx, cy)
      if ok then
        app.hint_move = nil
        app.ai_delay = 0.25
        app.engine:sync_position(app.board:to_uci_moves())
      end
    end
  end
end

function love.keypressed(key)
  if key == "escape" then
    app.show_perft = false
    app.show_about = false
    ui.active_menu = nil
  elseif key == "n" then
    app.handle_action("new_game")
  elseif key == "u" or key == "z" then
    app.handle_action("undo")
  elseif key == "y" then
    app.handle_action("redo")
  elseif key == "h" then
    app.handle_action("hint")
  elseif key == "a" then
    app.handle_action("analyze")
  elseif key == "p" then
    app.show_perft = true
  elseif key == "space" and app.board.forced_pass then
    app.handle_action("pass")
  end
end

function love.quit()
  if app.engine then
    app.engine:quit()
  end
end

return app

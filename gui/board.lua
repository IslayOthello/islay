-- Othello / Reversi Board & Game Logic for Islay (WZebra style)
-- Backed by canonical bitboard representation (islay::Board / islay::GameStatus)

local bitboard = require("gui.bitboard")
local openings = require("gui.openings")

local board = {}
board.__index = board

board.EMPTY = bitboard.EMPTY
board.BLACK = bitboard.BLACK
board.WHITE = bitboard.WHITE
board.PASS  = bitboard.PASS

function board.new(rule)
  local self = setmetatable({}, board)
  self.rule = rule or "Othello"
  self:reset()
  return self
end

function board:reset()
  self.rep = bitboard.Board.start()
  self.turn = board.BLACK
  self.grid = self.rep:to_grid(self.turn)

  self.disc_move_numbers = {}
  for y = 1, 8 do
    self.disc_move_numbers[y] = {}
    for x = 1, 8 do
      self.disc_move_numbers[y][x] = 0
    end
  end

  self.history = {}
  self.current_ply = 0
  self.last_move = nil
  self.initial_fen = nil
  self.initial_stm = nil
  self.animating_flips = {}
  self.game_over = false
  self.winner = nil
  self.forced_pass = false
  self.setup_mode = false
  self.legal_evals = {} -- [uci] -> eval score

  self:check_game_state()
end

function board:opponent(color)
  if color == board.BLACK then return board.WHITE end
  if color == board.WHITE then return board.BLACK end
  return board.EMPTY
end

function board:get(x, y)
  if x < 1 or x > 8 or y < 1 or y > 8 then return nil end
  return self.grid[y][x]
end

function board:get_move_number(x, y)
  if x < 1 or x > 8 or y < 1 or y > 8 then return 0 end
  return self.disc_move_numbers[y][x]
end

-- Verify whether a UCI move string ("c4", "pass") is legal in the current position
function board:verify_move(uci_str)
  local sq = bitboard.parse_square(uci_str)
  if sq == bitboard.NOMOVE then
    return false, "Unparseable square coordinate: " .. tostring(uci_str)
  end
  local ok, _, err = self.rep:try_play(sq, self.rule)
  return ok, err
end

-- Compute flipped discs for a move at (x, y)
function board:get_flips(x, y, color)
  color = color or self.turn
  local sq = bitboard.from_xy(x, y)
  if sq == bitboard.NOMOVE then return {} end

  local p_bb, o_bb
  if color == self.turn then
    p_bb = self.rep.player
    o_bb = self.rep.opponent
  else
    p_bb = self.rep.opponent
    o_bb = self.rep.player
  end

  local _, flips_list = bitboard.flip(sq, p_bb, o_bb)
  return flips_list
end

-- Get all legal moves as table [uci] -> { x, y, uci, flips }
function board:get_legal_moves(color)
  color = color or self.turn
  local moves = {}

  local p_bb, o_bb
  if color == self.turn then
    p_bb = self.rep.player
    o_bb = self.rep.opponent
  else
    p_bb = self.rep.opponent
    o_bb = self.rep.player
  end

  local moves_bb = bitboard.get_moves(p_bb, o_bb)
  for sq = 0, 63 do
    if bit.band(moves_bb, bitboard.square_bb(sq)) ~= 0ULL then
      local uci = bitboard.square_to_string(sq)
      local x, y = bitboard.to_xy(sq)
      local _, flips = bitboard.flip(sq, p_bb, o_bb)
      moves[uci] = {
        x = x,
        y = y,
        uci = uci,
        flips = flips,
        sq = sq
      }
    end
  end
  return moves
end

function board:has_legal_moves(color)
  color = color or self.turn
  if color == self.turn then
    return self.rep:has_moves()
  else
    return self.rep:passed():has_moves()
  end
end

function board.to_uci(x, y)
  if x == board.PASS and y == board.PASS then return "pass" end
  return bitboard.square_to_string(bitboard.from_xy(x, y))
end

function board.from_uci(s)
  local sq = bitboard.parse_square(s)
  if sq == bitboard.PASS then return board.PASS, board.PASS end
  if sq == bitboard.NOMOVE then return nil, nil end
  return bitboard.to_xy(sq)
end

function board:clone_grid()
  local g = {}
  for y = 1, 8 do
    g[y] = {}
    for x = 1, 8 do g[y][x] = self.grid[y][x] end
  end
  return g
end

function board:clone_numbers()
  local n = {}
  for y = 1, 8 do
    n[y] = {}
    for x = 1, 8 do n[y][x] = self.disc_move_numbers[y][x] end
  end
  return n
end

-- Apply user or engine move at (x, y) with rigorous correctness check
function board:apply_move(x, y, eval_score)
  if self.game_over then return false, "Game is already over" end
  if self.setup_mode then return false, "Cannot play while in position setup mode" end

  local sq = bitboard.from_xy(x, y)
  if sq == bitboard.NOMOVE then
    return false, string.format("Invalid coordinates: (%s, %s)", tostring(x), tostring(y))
  end

  -- Validate against canonical bitboard representation
  local ok, next_rep, err, f_bb, flips = self.rep:try_play(sq, self.rule)
  if not ok then
    return false, err or "Illegal move"
  end

  -- Truncate redo history if branching from an earlier ply
  if self.current_ply < #self.history then
    while #self.history > self.current_ply do
      table.remove(self.history)
    end
  end

  local uci = bitboard.square_to_string(sq)
  local move_no = #self.history + 1
  local moving_color = self.turn
  local opp_color = self:opponent(moving_color)

  -- Record complete history record
  table.insert(self.history, {
    ply = move_no,
    x = x,
    y = y,
    sq = sq,
    uci = uci,
    color = moving_color,
    flips = flips,
    eval_score = eval_score or 0.0,
    prev_rep = self.rep,
    prev_grid = self:clone_grid(),
    prev_numbers = self:clone_numbers(),
    prev_turn = self.turn,
    prev_forced_pass = self.forced_pass,
    prev_game_over = self.game_over
  })
  self.current_ply = #self.history

  -- Update canonical bitboard
  self.rep = next_rep

  -- Update 2D display grid and move numbers
  self.grid[y][x] = moving_color
  self.disc_move_numbers[y][x] = move_no

  self.animating_flips = {}
  for i, f in ipairs(flips) do
    self.grid[f.y][f.x] = moving_color
    table.insert(self.animating_flips, {
      x = f.x,
      y = f.y,
      from_color = opp_color,
      to_color = moving_color,
      delay = (i - 1) * 0.03,
      timer = 0,
      duration = 0.20,
      done = false
    })
  end

  self.last_move = { x = x, y = y, uci = uci, color = moving_color, flips = flips }
  self.turn = opp_color
  self.forced_pass = false
  self.legal_evals = {}

  self:check_game_state()
  return true
end

-- Pass turn with rigorous correctness check
function board:pass(eval_score)
  if self.game_over then return false, "Game is already over" end
  if self.setup_mode then return false, "Cannot pass in setup mode" end

  -- Validate against canonical bitboard representation
  local ok, next_rep, err = self.rep:try_play(bitboard.PASS, self.rule)
  if not ok then
    return false, err or "Pass is not permitted"
  end

  if self.current_ply < #self.history then
    while #self.history > self.current_ply do table.remove(self.history) end
  end

  local move_no = #self.history + 1
  local opp_color = self:opponent(self.turn)

  table.insert(self.history, {
    ply = move_no,
    x = board.PASS,
    y = board.PASS,
    sq = bitboard.PASS,
    uci = "pass",
    color = self.turn,
    flips = {},
    eval_score = eval_score or 0.0,
    prev_rep = self.rep,
    prev_grid = self:clone_grid(),
    prev_numbers = self:clone_numbers(),
    prev_turn = self.turn,
    prev_forced_pass = self.forced_pass,
    prev_game_over = self.game_over
  })
  self.current_ply = #self.history

  self.rep = next_rep
  self.last_move = { x = board.PASS, y = board.PASS, uci = "pass", color = self.turn }
  self.turn = opp_color
  self.forced_pass = false
  self.legal_evals = {}

  self:check_game_state()
  return true
end

function board:goto_ply(target_ply)
  if target_ply < 0 or target_ply > #self.history then return end
  if target_ply == self.current_ply then return end

  local saved_history = self.history
  local saved_rule = self.rule
  local saved_fen = self.initial_fen
  local saved_stm = self.initial_stm
  self:reset()
  self.rule = saved_rule
  self.history = saved_history
  if saved_fen then
    local rep = bitboard.Board.from_fen(saved_fen, saved_stm)
    if rep then
      self.initial_fen = saved_fen
      self.initial_stm = saved_stm
      self.turn = (saved_stm == "O" or saved_stm == "W") and board.WHITE or board.BLACK
      self.rep = rep
      self.grid = self.rep:to_grid(self.turn)
    end
  end

  for i = 1, target_ply do
    local m = self.history[i]
    if m.uci == "pass" then
      self.rep = self.rep:passed()
      self.turn = self:opponent(m.color)
      self.last_move = { x = board.PASS, y = board.PASS, uci = "pass", color = m.color }
    else
      local ok, next_rep, _, _, flips = self.rep:try_play(m.sq, self.rule)
      if ok then
        self.rep = next_rep
      else
        self.rep, _, flips = self.rep:play(m.sq)
      end
      self.grid[m.y][m.x] = m.color
      self.disc_move_numbers[m.y][m.x] = i
      for _, f in ipairs(flips or m.flips) do
        self.grid[f.y][f.x] = m.color
      end
      self.last_move = { x = m.x, y = m.y, uci = m.uci, color = m.color, flips = flips or m.flips }
      self.turn = self:opponent(m.color)
    end
  end

  self.current_ply = target_ply
  self.forced_pass = false
  self:check_game_state()
end

function board:can_undo() return self.current_ply > 0 end
function board:undo()
  if not self:can_undo() then return false end
  self:goto_ply(self.current_ply - 1)
  return true
end

function board:can_redo() return self.current_ply < #self.history end
function board:redo()
  if not self:can_redo() then return false end
  self:goto_ply(self.current_ply + 1)
  return true
end

function board:goto_start() self:goto_ply(0) end
function board:goto_end() self:goto_ply(#self.history) end

-- Reconcile game state using canonical GameStatus
function board:check_game_state()
  local st = self.rep:status(self.rule)
  if st.terminal then
    self.game_over = true
    self.forced_pass = false
    self:finalize_winner()
  elseif st.forced_pass then
    self.game_over = false
    self.forced_pass = true
  else
    self.game_over = false
    self.forced_pass = false
  end
end

function board:finalize_winner()
  local b, w = self:get_counts()
  if b > w then self.winner = board.BLACK
  elseif w > b then self.winner = board.WHITE
  else self.winner = 0 end -- Draw
end

function board:get_counts()
  local p_cnt, o_cnt, e_cnt = self.rep:counts()
  local b_cnt = (self.turn == board.BLACK) and p_cnt or o_cnt
  local w_cnt = (self.turn == board.BLACK) and o_cnt or p_cnt
  return b_cnt, w_cnt, e_cnt
end

function board:to_uci_moves()
  local list = {}
  for i = 1, self.current_ply do
    table.insert(list, self.history[i].uci)
  end
  return table.concat(list, " ")
end

function board:get_opening_name()
  return openings.identify(self:to_uci_moves())
end

-- Position Setup Mode Helpers
function board:rebuild_rep_from_grid()
  local black = 0ULL
  local white = 0ULL
  for y = 1, 8 do
    for x = 1, 8 do
      local sq = bitboard.from_xy(x, y)
      local b = bitboard.square_bb(sq)
      if self.grid[y][x] == board.BLACK then
        black = bit.bor(black, b)
      elseif self.grid[y][x] == board.WHITE then
        white = bit.bor(white, b)
      end
    end
  end

  if self.turn == board.BLACK then
    self.rep = bitboard.new_board(black, white)
  else
    self.rep = bitboard.new_board(white, black)
  end
  self.history = {}
  self.current_ply = 0
  self.last_move = nil
  self.forced_pass = false
  self.game_over = false
  self.winner = nil
  self.initial_fen, self.initial_stm = self:to_fen()
  self:check_game_state()
end

function board:set_piece(x, y, color)
  if x >= 1 and x <= 8 and y >= 1 and y <= 8 then
    self.grid[y][x] = color
    self.disc_move_numbers[y][x] = 0
    self:rebuild_rep_from_grid()
  end
end

function board:clear_board()
  for y = 1, 8 do
    for x = 1, 8 do
      self.grid[y][x] = board.EMPTY
      self.disc_move_numbers[y][x] = 0
    end
  end
  self.rep = bitboard.new_board(0ULL, 0ULL)
  self.history = {}
  self.current_ply = 0
  self.last_move = nil
  self.game_over = false
  self.winner = nil
  self.forced_pass = false
  self.initial_fen, self.initial_stm = self:to_fen()
end

function board:invert_board()
  for y = 1, 8 do
    for x = 1, 8 do
      if self.grid[y][x] == board.BLACK then
        self.grid[y][x] = board.WHITE
      elseif self.grid[y][x] == board.WHITE then
        self.grid[y][x] = board.BLACK
      end
    end
  end
  self.turn = self:opponent(self.turn)
  self:rebuild_rep_from_grid()
end

function board:to_fen()
  local stm = (self.turn == board.BLACK) and "X" or "O"
  return self.rep:to_fen(stm)
end

function board:load_fen(fen_str, stm)
  local new_rep, err = bitboard.Board.from_fen(fen_str, stm)
  if not new_rep then return false, err end

  self.turn = (stm == "O" or stm == "W") and board.WHITE or board.BLACK
  self.rep = new_rep
  self.grid = self.rep:to_grid(self.turn)

  for y = 1, 8 do
    self.disc_move_numbers[y] = {}
    for x = 1, 8 do
      self.disc_move_numbers[y][x] = 0
    end
  end

  self.history = {}
  self.current_ply = 0
  self.last_move = nil
  self.forced_pass = false
  self.initial_fen, self.initial_stm = self:to_fen()
  self:check_game_state()
  return true
end

function board:update_animations(dt)
  for _, anim in ipairs(self.animating_flips) do
    if not anim.done then
      if anim.delay > 0 then
        anim.delay = anim.delay - dt
      else
        anim.timer = anim.timer + dt
        anim.progress = math.min(1, anim.timer / anim.duration)
        if anim.progress >= 1 then anim.done = true end
      end
    end
  end
end

function board:is_animating()
  for _, anim in ipairs(self.animating_flips) do
    if not anim.done then return true end
  end
  return false
end

return board

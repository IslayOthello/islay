-- Ported Board Representation & Movegen for Islay (WZebra style)
-- Faithful bitboard implementation of islay::Board and islay::GameStatus

local bit = require("bit")

local bitboard = {}

bitboard.PASS   = 64
bitboard.NOMOVE = 65

bitboard.EMPTY = 0
bitboard.BLACK = 1
bitboard.WHITE = 2

-- Square bitmask (0..63)
function bitboard.square_bb(sq)
  if sq < 0 or sq >= 64 then return 0ULL end
  return bit.lshift(1ULL, sq)
end

-- 64-bit Hamming weight (popcount)
function bitboard.popcount(b)
  local m1 = 0x5555555555555555ULL
  local m2 = 0x3333333333333333ULL
  local m4 = 0x0F0F0F0F0F0F0F0FULL
  b = b - bit.band(bit.rshift(b, 1), m1)
  b = bit.band(b, m2) + bit.band(bit.rshift(b, 2), m2)
  b = bit.band(b + bit.rshift(b, 4), m4)
  b = b * 0x0101010101010101ULL
  return tonumber(bit.rshift(b, 56))
end

-- Convert 1-based (x, y) to rank-major square (0..63)
-- x: 1..8 (file a..h), y: 1..8 (rank 1..8)
function bitboard.from_xy(x, y)
  if not x or not y or x < 1 or x > 8 or y < 1 or y > 8 then
    return bitboard.NOMOVE
  end
  return (y - 1) * 8 + (x - 1)
end

-- Convert square (0..63) to 1-based (x, y)
function bitboard.to_xy(sq)
  if sq == bitboard.PASS then return bitboard.PASS, bitboard.PASS end
  if sq < 0 or sq >= 64 then return nil, nil end
  return (sq % 8) + 1, math.floor(sq / 8) + 1
end

-- Parse string ("c4", "pass", "--") to Square index (0..64)
function bitboard.parse_square(s)
  if not s or s == "" then return bitboard.NOMOVE end
  s = s:lower():gsub("%s+", "")
  if s == "pass" or s == "--" or s == "@@" then return bitboard.PASS end
  if #s < 2 then return bitboard.NOMOVE end
  local fc = string.byte(s:sub(1, 1)) - 97 -- 'a' -> 0 .. 'h' -> 7
  local rc = string.byte(s:sub(2, 2)) - 49 -- '1' -> 0 .. '8' -> 7
  if fc < 0 or fc > 7 or rc < 0 or rc > 7 then return bitboard.NOMOVE end
  return rc * 8 + fc
end

-- Convert square index (0..64) to string ("c4", "pass")
function bitboard.square_to_string(sq)
  if sq == bitboard.PASS then return "pass" end
  if sq < 0 or sq >= 64 then return "--" end
  local fc = string.char(97 + (sq % 8))
  local rc = string.char(49 + math.floor(sq / 8))
  return fc .. rc
end

-- 8 ray directions (dx, dy)
local DIRS = {
  { 1,  0}, { 0,  1}, {-1,  0}, { 0, -1},
  { 1,  1}, {-1,  1}, { 1, -1}, {-1, -1}
}

-- Calculate flipped discs bitmask for a move at sq
-- Returns (flips_bitboard, flips_list)
function bitboard.flip(sq, player, opponent)
  if sq < 0 or sq >= 64 then return 0ULL, {} end
  local occ = bit.bor(player, opponent)
  if bit.band(occ, bitboard.square_bb(sq)) ~= 0ULL then
    return 0ULL, {} -- Square already occupied
  end

  local x0 = sq % 8
  local y0 = math.floor(sq / 8)
  local flips_bb = 0ULL
  local flips_list = {}

  for _, d in ipairs(DIRS) do
    local dx, dy = d[1], d[2]
    local x = x0 + dx
    local y = y0 + dy
    local ray = 0ULL
    local ray_list = {}

    while x >= 0 and x < 8 and y >= 0 and y < 8 do
      local b = bitboard.square_bb(y * 8 + x)
      if bit.band(opponent, b) == 0ULL then
        if bit.band(player, b) ~= 0ULL then
          -- Ray is bounded by mover's disc: all opponent discs in ray are flipped!
          flips_bb = bit.bor(flips_bb, ray)
          for _, pt in ipairs(ray_list) do
            table.insert(flips_list, pt)
          end
        end
        break
      end
      ray = bit.bor(ray, b)
      table.insert(ray_list, { x = x + 1, y = y + 1, sq = y * 8 + x })
      x = x + dx
      y = y + dy
    end
  end

  return flips_bb, flips_list
end

-- Compute legal moves bitboard for current mover
function bitboard.get_moves(player, opponent)
  local moves = 0ULL
  local occ = bit.bor(player, opponent)
  for sq = 0, 63 do
    if bit.band(occ, bitboard.square_bb(sq)) == 0ULL then
      local f = bitboard.flip(sq, player, opponent)
      if f ~= 0ULL then
        moves = bit.bor(moves, bitboard.square_bb(sq))
      end
    end
  end
  return moves
end

function bitboard.can_move(player, opponent)
  return bitboard.get_moves(player, opponent) ~= 0ULL
end

-- --- Board Representation (Mover-Relative) ---
local Board = {}
Board.__index = Board

function bitboard.new_board(player, opponent)
  local self = setmetatable({}, Board)
  self.player = player or 0ULL
  self.opponent = opponent or 0ULL
  return self
end

-- Standard tournament start position:
-- d4=27, e5=36: White; e4=28, d5=35: Black
-- Mover is Black.
function Board.start()
  local black = bit.bor(bitboard.square_bb(28), bitboard.square_bb(35))
  local white = bit.bor(bitboard.square_bb(27), bitboard.square_bb(36))
  return bitboard.new_board(black, white)
end

-- Parse diagram string (64 characters) and side-to-move ('X' or 'O')
function Board.from_fen(diagram, stm)
  diagram = diagram:gsub("%s+", "")
  if #diagram < 64 then return nil, "FEN diagram must have at least 64 characters" end
  local black = 0ULL
  local white = 0ULL
  for sq = 0, 63 do
    local ch = diagram:sub(sq + 1, sq + 1):upper()
    if ch == "X" or ch == "*" then
      black = bit.bor(black, bitboard.square_bb(sq))
    elseif ch == "O" or ch == "0" then
      white = bit.bor(white, bitboard.square_bb(sq))
    elseif ch == "-" or ch == "." or ch == "_" then
      -- empty
    else
      return nil, "Invalid diagram character: " .. ch
    end
  end

  if bit.band(black, white) ~= 0ULL then
    return nil, "Overlapping discs in diagram"
  end

  stm = stm and stm:upper() or "X"
  if stm == "O" or stm == "W" then
    return bitboard.new_board(white, black), nil -- White to move: player is white
  else
    return bitboard.new_board(black, white), nil -- Black to move: player is black
  end
end

-- Export diagram string (64 characters)
function Board:to_fen(stm)
  local is_black_stm = (stm == "X" or stm == "B" or stm == bitboard.BLACK)
  local black = is_black_stm and self.player or self.opponent
  local white = is_black_stm and self.opponent or self.player

  local chars = {}
  for sq = 0, 63 do
    local b = bitboard.square_bb(sq)
    if bit.band(black, b) ~= 0ULL then
      chars[sq + 1] = "X"
    elseif bit.band(white, b) ~= 0ULL then
      chars[sq + 1] = "O"
    else
      chars[sq + 1] = "-"
    end
  end
  return table.concat(chars), is_black_stm and "X" or "O"
end

function Board:moves()
  return bitboard.get_moves(self.player, self.opponent)
end

function Board:has_moves()
  return bitboard.can_move(self.player, self.opponent)
end

function Board:passed()
  return bitboard.new_board(self.opponent, self.player)
end

function Board:play(sq)
  local f, flips_list = bitboard.flip(sq, self.player, self.opponent)
  local next_p = bit.bxor(self.opponent, f)
  local next_o = bit.bxor(self.player, bit.bor(f, bitboard.square_bb(sq)))
  return bitboard.new_board(next_p, next_o), f, flips_list
end

function Board:count()
  return bitboard.popcount(bit.bor(self.player, self.opponent))
end

function Board:counts()
  local p_cnt = bitboard.popcount(self.player)
  local o_cnt = bitboard.popcount(self.opponent)
  local e_cnt = 64 - (p_cnt + o_cnt)
  return p_cnt, o_cnt, e_cnt
end

-- Detailed game status check
function Board:status(rule)
  rule = rule or "Othello"
  local moves = self:moves()
  if moves ~= 0ULL then
    return {
      moves = moves,
      forced_pass = false,
      terminal = false,
      action_count = bitboard.popcount(moves),
      margin = bitboard.popcount(self.player) - bitboard.popcount(self.opponent)
    }
  end

  -- No legal moves for current mover
  if rule == "Othello" then
    local opp_moves = self:passed():moves()
    if opp_moves ~= 0ULL then
      -- Mover has 0 moves, but opponent has legal moves -> forced pass!
      return {
        moves = 0ULL,
        forced_pass = true,
        terminal = false,
        action_count = 1,
        margin = bitboard.popcount(self.player) - bitboard.popcount(self.opponent)
      }
    end
  end

  -- Game is over (terminal): neither player can move, or Reversi stuck
  local p_cnt = bitboard.popcount(self.player)
  local o_cnt = bitboard.popcount(self.opponent)
  local winner = nil
  if p_cnt > o_cnt then winner = 1
  elseif o_cnt > p_cnt then winner = 2
  else winner = 0 end -- Draw

  return {
    moves = 0ULL,
    forced_pass = false,
    terminal = true,
    action_count = 0,
    winner = winner,
    margin = p_cnt - o_cnt
  }
end

-- Validate and play an action (square 0..63 or PASS)
-- Returns: (ok, next_board, err_message, flips_bb, flips_list)
function Board:try_play(action, rule)
  rule = rule or "Othello"

  -- 1. Sanity check disjoint bitboards
  if bit.band(self.player, self.opponent) ~= 0ULL then
    return false, self, "Corrupted board: overlapping discs between player and opponent"
  end

  local st = self:status(rule)
  if st.terminal then
    return false, self, "Game is already over (terminal state reached)"
  end

  -- 2. Handle PASS
  if action == bitboard.PASS then
    if not st.forced_pass then
      if st.moves ~= 0ULL then
        return false, self, "Cannot pass: legal moves are available (" .. bitboard.popcount(st.moves) .. " candidate squares)"
      else
        return false, self, "Cannot pass: rule " .. rule .. " ends game on no moves"
      end
    end
    return true, self:passed(), nil, 0ULL, {}
  end

  -- 3. Handle square action (0..63)
  if type(action) ~= "number" or action < 0 or action >= 64 then
    return false, self, "Invalid action index: " .. tostring(action)
  end

  if st.forced_pass then
    return false, self, "Must pass: no legal moves available on the board"
  end

  local action_bb = bitboard.square_bb(action)
  if bit.band(st.moves, action_bb) == 0ULL then
    local occ = bit.bor(self.player, self.opponent)
    if bit.band(occ, action_bb) ~= 0ULL then
      return false, self, "Square " .. bitboard.square_to_string(action):upper() .. " is already occupied"
    else
      return false, self, "Square " .. bitboard.square_to_string(action):upper() .. " does not flip any opponent discs"
    end
  end

  -- Move is fully verified and legal!
  local next_board, f_bb, f_list = self:play(action)
  return true, next_board, nil, f_bb, f_list
end

-- Convert to 2D grid: 1 = Black, 2 = White, 0 = Empty
function Board:to_grid(stm)
  local is_b = (stm == "X" or stm == "B" or stm == bitboard.BLACK)
  local black = is_b and self.player or self.opponent
  local white = is_b and self.opponent or self.player

  local grid = {}
  for y = 1, 8 do
    grid[y] = {}
    for x = 1, 8 do
      local sq = (y - 1) * 8 + (x - 1)
      local b = bitboard.square_bb(sq)
      if bit.band(black, b) ~= 0ULL then
        grid[y][x] = bitboard.BLACK
      elseif bit.band(white, b) ~= 0ULL then
        grid[y][x] = bitboard.WHITE
      else
        grid[y][x] = bitboard.EMPTY
      end
    end
  end
  return grid
end

-- Built-in perft calculation for exact self-testing
function bitboard.perft(board, depth, rule)
  rule = rule or "Othello"
  if depth <= 0 then return 1 end
  local st = board:status(rule)
  if st.terminal then return 0 end

  local total = 0
  if st.forced_pass then
    local ok, nxt = board:try_play(bitboard.PASS, rule)
    if ok then
      total = total + bitboard.perft(nxt, depth - 1, rule)
    end
  else
    for sq = 0, 63 do
      if bit.band(st.moves, bitboard.square_bb(sq)) ~= 0ULL then
        local ok, nxt = board:try_play(sq, rule)
        if ok then
          total = total + bitboard.perft(nxt, depth - 1, rule)
        end
      end
    end
  end
  return total
end

-- Self-test against known perft numbers
function bitboard.selftest()
  local start = Board.start()
  local expected = {
    [1] = 4,
    [2] = 12,
    [3] = 56,
    [4] = 244,
    [5] = 1396
  }

  for d = 1, 4 do
    local n = bitboard.perft(start, d, "Othello")
    if n ~= expected[d] then
      return false, string.format("Perft(%d) mismatch: expected %d, got %d", d, expected[d], n)
    end
  end

  -- Verify try_play rejection on illegal move
  local ok, _, err = start:try_play(0, "Othello") -- a1 is empty but does not flip
  if ok then return false, "Illegal move a1 was accepted" end

  local ok2, _, _ = start:try_play(bitboard.PASS, "Othello") -- pass when moves available
  if ok2 then return false, "Illegal pass was accepted at startpos" end

  local ok3, nxt, _, _, flips = start:try_play(26, "Othello") -- c4 is legal
  if not ok3 or #flips ~= 1 then return false, "Legal move c4 failed" end

  return true, "All bitboard selftests passed"
end

bitboard.Board = Board
return bitboard

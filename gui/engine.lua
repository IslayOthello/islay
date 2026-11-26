-- UCI Protocol Communicator for Islay Engine
local engine = {}
engine.__index = engine

local has_ffi, ffi = pcall(require, "ffi")

if has_ffi then
  pcall(function()
    ffi.cdef[[
      struct pollfd {
        int fd;
        short events;
        short revents;
      };
      int poll(struct pollfd *fds, unsigned int nfds, int timeout);
      int pipe(int pipefd[2]);
      int fork(void);
      int dup2(int oldfd, int newfd);
      int close(int fd);
      int execl(const char *path, const char *arg0, ...);
      ssize_t read(int fd, void *buf, size_t count);
      ssize_t write(int fd, const void *buf, size_t count);
      int kill(int pid, int sig);
      int waitpid(int pid, int *status, int options);
      int access(const char *pathname, int mode);
    ]]
  end)
end

local function find_binary(preferred)
  local candidates = {}
  if preferred and preferred ~= "" then
    table.insert(candidates, preferred)
  end
  table.insert(candidates, "./build/islay")
  table.insert(candidates, "../build/islay")
  table.insert(candidates, "./_build/islay")
  table.insert(candidates, "../_build/islay")
  table.insert(candidates, "./islay")
  table.insert(candidates, "../islay")
  for _, path in ipairs(candidates) do
    local f = io.open(path, "rb")
    if f then
      local byte = f:read(1)
      f:close()
      if byte and (not has_ffi or ffi.C.access(path, 1) == 0) then
        return path
      end
    end
  end
  return nil
end

local function find_eval_file()
  local candidates = {
    "./wdl-gen5.onnx", "../wdl-gen5.onnx",
    "./wdl-gen4.onnx", "../wdl-gen4.onnx",
    "./wdl-gen3.onnx", "../wdl-gen3.onnx",
    "./wdl-gen2.onnx", "../wdl-gen2.onnx",
    "./wdl-gen1.onnx", "../wdl-gen1.onnx",
  }
  for _, path in ipairs(candidates) do
    local f = io.open(path, "rb")
    if f then
      local byte = f:read(1)
      f:close()
      if byte then return path end
    end
  end
  return nil
end

function engine.new(binary_path)
  local self = setmetatable({}, engine)
  self.binary_path = find_binary(binary_path)
  self.state = "disconnected" -- "disconnected", "ready", "searching", "perft"
  self.pid = nil
  self.in_fd = nil
  self.out_fd = nil
  self.pfd = nil
  self.read_buffer = ""
  self.pending_searches = 0

  self.stats = {
    nodes = 0,
    nps = 0,
    time = 0,
    pv = "",
    value = 0,
    evaluations = 0,
    reason = "",
    bestmove = nil,
    evaluator = "uniform",
    exact_wdl = nil,
    candidates = {} -- list of { move, visits, pct, value, prior }
  }

  self.perft_results = {
    depth = 0,
    branches = {},
    nodes = 0,
    time_str = "",
    speed_str = ""
  }

  self.on_ready = nil
  self.on_info = nil
  self.on_bestmove = nil
  self.on_perft_done = nil
  self.on_error = nil

  return self
end

function engine:is_connected()
  return self.state ~= "disconnected" and self.pid ~= nil
end

function engine:start()
  if self:is_connected() then return true end
  if not has_ffi then
    if self.on_error then self.on_error("LuaJIT FFI required for engine IPC") end
    return false
  end

  self.binary_path = find_binary(self.binary_path)
  if not self.binary_path then
    if self.on_error then self.on_error("Could not find islay binary. Build with 'cmake --build build' first.") end
    return false
  end

  local in_pipe = ffi.new("int[2]")
  local out_pipe = ffi.new("int[2]")
  if ffi.C.pipe(in_pipe) ~= 0 or ffi.C.pipe(out_pipe) ~= 0 then
    if self.on_error then self.on_error("Failed to create POSIX pipes") end
    return false
  end

  local pid = ffi.C.fork()
  if pid == 0 then
    -- Child process
    ffi.C.dup2(in_pipe[0], 0)
    ffi.C.dup2(out_pipe[1], 1)
    ffi.C.close(in_pipe[0])
    ffi.C.close(in_pipe[1])
    ffi.C.close(out_pipe[0])
    ffi.C.close(out_pipe[1])

    ffi.C.execl(self.binary_path, self.binary_path, ffi.cast("char*", nil))
    os.exit(127)
  end

  -- Parent process
  ffi.C.close(in_pipe[0])
  ffi.C.close(out_pipe[1])

  self.pid = pid
  self.in_fd = in_pipe[1]
  self.out_fd = out_pipe[0]

  self.pfd = ffi.new("struct pollfd[1]")
  self.pfd[0].fd = self.out_fd
  self.pfd[0].events = 1 -- POLLIN

  self.read_buffer = ""
  self.pending_searches = 0
  self.state = "initializing"

  -- Send initialization handshake
  self:send("uci")
  self:set_mcts_hash(512)
  self:set_mcts_batch(16)
  local eval_path = find_eval_file()
  if eval_path then
    self:set_eval_file(eval_path)
  end
  self:send("isready")
  return true
end

function engine:send(cmd)
  if not self.in_fd then return false end
  local msg = cmd .. "\n"
  local written = ffi.C.write(self.in_fd, msg, #msg)
  return written == #msg
end

function engine:set_rule(rule)
  self:send("setoption name Rule value " .. (rule or "Othello"))
end

function engine:set_eval_file(path)
  self.eval_file = path or ""
  self:send("setoption name EvalFile value " .. (path and path ~= "" and path or "<empty>"))
end

function engine:set_mcts_hash(mib)
  self:send("setoption name MctsHash value " .. tostring(mib or 512))
end

function engine:set_mcts_batch(batch)
  self:send("setoption name MctsBatch value " .. tostring(batch or 16))
end

function engine:set_q_aware_stop(enabled)
  self:send("setoption name MctsQAwareStop value " .. (enabled and "true" or "false"))
end

function engine:set_perft_hash(mib)
  self:send("setoption name PerftHash value " .. tostring(mib or 512))
end

function engine:sync_position(moves_str, fen_str, stm)
  if fen_str and #fen_str == 64 then
    local cmd = "position fen " .. fen_str .. " " .. (stm or "X")
    if moves_str and moves_str ~= "" then
      cmd = cmd .. " moves " .. moves_str
    end
    self:send(cmd)
  else
    if moves_str and moves_str ~= "" then
      self:send("position startpos moves " .. moves_str)
    else
      self:send("position startpos")
    end
  end
end

function engine:search_nodes(nodes)
  if not self:is_connected() then return false end
  self.stats.bestmove = nil
  self.stats.reason = ""
  self.stats.exact_wdl = nil
  self.stats.value = nil
  self.stats.candidates = {}
  self.pending_searches = (self.pending_searches or 0) + 1
  self.state = "searching"
  return self:send("go nodes " .. tostring(nodes or 1000))
end

function engine:search_movetime(ms)
  if not self:is_connected() then return false end
  self.stats.bestmove = nil
  self.stats.reason = ""
  self.stats.exact_wdl = nil
  self.stats.value = nil
  self.stats.candidates = {}
  self.pending_searches = (self.pending_searches or 0) + 1
  self.state = "searching"
  return self:send("go movetime " .. tostring(ms or 500))
end

function engine:search_infinite()
  if not self:is_connected() then return false end
  self.stats.bestmove = nil
  self.stats.reason = ""
  self.stats.exact_wdl = nil
  self.stats.value = nil
  self.stats.candidates = {}
  self.pending_searches = (self.pending_searches or 0) + 1
  self.state = "searching"
  return self:send("go infinite")
end

function engine:perft(depth)
  if not self:is_connected() then return false end
  self.perft_results = {
    depth = depth,
    branches = {},
    nodes = 0,
    time_str = "",
    speed_str = ""
  }
  self.state = "perft"
  return self:send("go perft " .. tostring(depth or 5))
end

function engine:stop()
  if self.state == "searching" then
    self:send("stop")
  end
end

function engine:quit()
  if not self:is_connected() then return end
  self:send("quit")
  if self.in_fd then ffi.C.close(self.in_fd); self.in_fd = nil end
  if self.out_fd then ffi.C.close(self.out_fd); self.out_fd = nil end
  if self.pid then
    local status = ffi.new("int[1]")
    ffi.C.waitpid(self.pid, status, 0)
    self.pid = nil
  end
  self.state = "disconnected"
end

function engine:read_lines()
  if not self.out_fd or not self.pfd then return {} end

  local lines = {}
  while true do
    local r = ffi.C.poll(self.pfd, 1, 0) -- non-blocking check
    if r <= 0 or bit.band(self.pfd[0].revents, 1) == 0 then
      break
    end

    local buf = ffi.new("char[4096]")
    local n = tonumber(ffi.C.read(self.out_fd, buf, 4095))
    if n <= 0 then break end

    self.read_buffer = self.read_buffer .. ffi.string(buf, n)
    while true do
      local nl = self.read_buffer:find("\n")
      if not nl then break end
      local line = self.read_buffer:sub(1, nl - 1):gsub("\r", "")
      self.read_buffer = self.read_buffer:sub(nl + 1)
      table.insert(lines, line)
    end
  end

  return lines
end

function engine:update()
  if not self:is_connected() then return end

  local lines = self:read_lines()
  for _, line in ipairs(lines) do
    self:handle_line(line)
  end
end

function engine:handle_line(line)
  if line == "readyok" then
    if (self.pending_searches or 0) == 0 then
      self.state = "ready"
    end
    if self.on_ready then self.on_ready() end

  elseif line:find("^info error:") then
    local err = line:match("^info error:%s*(.+)") or line
    if self.on_error then self.on_error(err) end

  elseif line:find("^info string evaluator") then
    self.stats.evaluator = line:match("^info string evaluator%s*(.+)") or "uniform"
    if self.on_info then self.on_info(self.stats) end

  elseif line:find("^info string option EvalFile =") then
    local val = line:match("^info string option EvalFile =%s*(.+)") or ""
    if val == "" or val == "<empty>" then
      self.eval_file = ""
      self.stats.evaluator = "uniform"
    else
      self.eval_file = val
      self.stats.evaluator = val:match("([^/\\]+)$") or val
    end
    if self.on_info then self.on_info(self.stats) end

  elseif line:find("^info nodes") then
    if (self.pending_searches or 0) > 1 then return end
    -- Parse: info nodes <N> nps <NPS> time <MS> [pv <moves>]
    local nodes = line:match("nodes%s+(%d+)")
    local nps = line:match("nps%s+(%d+)")
    local t = line:match("time%s+(%d+)")
    local pv = line:match("pv%s+(.+)")

    if nodes then self.stats.nodes = tonumber(nodes) end
    if nps then
      self.stats.nps = tonumber(nps)
    elseif nodes and t and tonumber(t) > 0 then
      self.stats.nps = math.floor(tonumber(nodes) * 1000 / tonumber(t))
    end
    if t then self.stats.time = tonumber(t) end
    if pv then self.stats.pv = pv end
    if self.on_info then self.on_info(self.stats) end

  elseif line:find("^info string search progress Q") then
    if (self.pending_searches or 0) > 1 then return end
    local val = line:match("Q%s+([%+%-%d%.]+)")
    if val then self.stats.value = tonumber(val) or 0 end
    if self.on_info then self.on_info(self.stats) end

  elseif line:find("^info string search policy:") then
    if (self.pending_searches or 0) > 1 then return end
    -- Parse: info string search policy: d3=45.2%/N452/Q+0.123 ...
    local cands = {}
    for mv, pct, vis, q in line:gmatch("([a-h][1-8])=([%d%.]+)%%/N(%d+)/Q([%+%-%d%.]+)") do
      table.insert(cands, {
        move = mv,
        pct = tonumber(pct) or 0,
        visits = tonumber(vis) or 0,
        value = tonumber(q) or 0,
        prior = 0
      })
    end
    if #cands > 0 then
      self.stats.candidates = cands
      if self.on_info then self.on_info(self.stats) end
    end

  elseif line:find("^info string candidate moves:") then
    if (self.pending_searches or 0) > 1 then return end
    self.stats.candidates = {}

  elseif line:find("^info string%s+#%d+") then
    if (self.pending_searches or 0) > 1 then return end
    -- Parse: info string   #1 d3: N=   452 (45.2%) Q=+0.123 P= 38.4%
    local mv, vis, pct, q, prior = line:match("#%d+%s+(%S+):%s+N=%s*(%d+)%s+%(%s*([%d%.]+)%%%)%s+Q=([%+%-%d%.]+)%s+P=%s*([%d%.]+)%%")
    if mv then
      table.insert(self.stats.candidates, {
        move = mv,
        visits = tonumber(vis) or 0,
        pct = tonumber(pct) or 0,
        value = tonumber(q) or 0,
        prior = tonumber(prior) or 0
      })
      if self.on_info then self.on_info(self.stats) end
    end

  elseif line:find("^info string exact WDL win/draw/loss") then
    if (self.pending_searches or 0) > 1 then return end
    self.stats.exact_wdl = line:match("win/draw/loss%s+(%S+)")
    if self.on_info then self.on_info(self.stats) end

  elseif line:find("^info string search") then
    if (self.pending_searches or 0) > 1 then return end
    -- Parse: info string search <reason> Q <V> ... evaluations <E>
    local reason = line:match("search%s+(%S+)")
    local val = line:match("Q%s+([%+%-%d%.]+)") or line:match("value%s+([%+%-%d%.]+)")
    local ev = line:match("evaluations%s+(%d+)")

    if reason then self.stats.reason = reason end
    if val then self.stats.value = tonumber(val) or 0 end
    if ev then self.stats.evaluations = tonumber(ev) end
    if self.on_info then self.on_info(self.stats) end

  elseif line:find("^bestmove") then
    self.pending_searches = math.max(0, (self.pending_searches or 1) - 1)
    if self.pending_searches > 0 then
      -- Stale bestmove from an earlier stopped search; keep waiting for active search
      self.stats.candidates = {}
      return
    end
    local bm = line:match("^bestmove%s+(%S+)")
    self.stats.bestmove = bm
    self.state = "ready"
    if self.on_bestmove then
      self.on_bestmove(bm, self.stats)
    end

  elseif self.state == "perft" then
    -- Branch count lines: "d3: 97554"
    local move, count = line:match("^([a-h][1-8]):%s+(%d+)")
    if move and count then
      table.insert(self.perft_results.branches, { move = move, count = tonumber(count) })
    elseif line:find("^Nodes searched:") then
      local n = line:match("Nodes searched:%s*([%d,]+)")
      if n then
        self.perft_results.nodes = tonumber(n:gsub(",", ""))
      end
    elseif line:find("^Time:") then
      self.perft_results.time_str = line:match("Time:%s*(.+)") or ""
    elseif line:find("^Speed:") then
      self.perft_results.speed_str = line:match("Speed:%s*(.+)") or ""
      self.state = "ready"
      if self.on_perft_done then
        self.on_perft_done(self.perft_results)
      end
    end
  end
end

return engine

-- Lightweight, pure-Lua JSON encoder and decoder
local json = {}

local function escape_str(s)
  local in_char  = {'\\', '"', '/', '\b', '\f', '\n', '\r', '\t'}
  local out_char = {'\\\\', '\\"', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t'}
  for i, c in ipairs(in_char) do
    s = s:gsub(c, out_char[i])
  end
  return '"' .. s .. '"'
end

function json.encode(val)
  local t = type(val)
  if t == "nil" then
    return "null"
  elseif t == "boolean" then
    return val and "true" or "false"
  elseif t == "number" then
    if val ~= val then return "null" end -- NaN
    if val >= math.huge or val <= -math.huge then return "null" end
    return tostring(val)
  elseif t == "string" then
    return escape_str(val)
  elseif t == "table" then
    local is_array = true
    local max_idx = 0
    for k, _ in pairs(val) do
      if type(k) == "number" and k > 0 and math.floor(k) == k then
        if k > max_idx then max_idx = k end
      else
        is_array = false
        break
      end
    end

    if is_array and max_idx > 0 then
      local parts = {}
      for i = 1, max_idx do
        table.insert(parts, json.encode(val[i]))
      end
      return "[" .. table.concat(parts, ",") .. "]"
    elseif is_array and max_idx == 0 and next(val) == nil then
      return "[]"
    else
      local parts = {}
      for k, v in pairs(val) do
        table.insert(parts, escape_str(tostring(k)) .. ":" .. json.encode(v))
      end
      return "{" .. table.concat(parts, ",") .. "}"
    end
  end
  return "null"
end

function json.decode(str)
  if not str or str == "" then return nil end
  -- Clean, safe JSON parser for common JSON types
  local pos = 1
  local len = #str

  local function skip_whitespace()
    while pos <= len do
      local b = str:byte(pos)
      if b == 32 or b == 9 or b == 10 or b == 13 then
        pos = pos + 1
      else
        break
      end
    end
  end

  local parse_val

  local function parse_str()
    pos = pos + 1 -- skip opening quote
    local start_p = pos
    local buf = {}
    while pos <= len do
      local c = str:sub(pos, pos)
      if c == '"' then
        pos = pos + 1
        return table.concat(buf)
      elseif c == '\\' then
        pos = pos + 1
        local esc = str:sub(pos, pos)
        if esc == 'n' then table.insert(buf, '\n')
        elseif esc == 'r' then table.insert(buf, '\r')
        elseif esc == 't' then table.insert(buf, '\t')
        elseif esc == '"' then table.insert(buf, '"')
        elseif esc == '\\' then table.insert(buf, '\\')
        else table.insert(buf, esc) end
        pos = pos + 1
      else
        table.insert(buf, c)
        pos = pos + 1
      end
    end
    return table.concat(buf)
  end

  local function parse_num()
    local s, e = str:find("^[%-%+]?[%d%.]+[eE]?[%-%+]?%d*", pos)
    if s then
      pos = e + 1
      return tonumber(str:sub(s, e))
    end
    return nil
  end

  local function parse_arr()
    pos = pos + 1 -- skip '['
    local arr = {}
    skip_whitespace()
    if pos <= len and str:sub(pos, pos) == ']' then
      pos = pos + 1
      return arr
    end
    while pos <= len do
      local v = parse_val()
      table.insert(arr, v)
      skip_whitespace()
      local c = str:sub(pos, pos)
      if c == ']' then
        pos = pos + 1
        return arr
      elseif c == ',' then
        pos = pos + 1
      else
        break
      end
    end
    return arr
  end

  local function parse_obj()
    pos = pos + 1 -- skip '{'
    local obj = {}
    skip_whitespace()
    if pos <= len and str:sub(pos, pos) == '}' then
      pos = pos + 1
      return obj
    end
    while pos <= len do
      skip_whitespace()
      if str:sub(pos, pos) ~= '"' then break end
      local k = parse_str()
      skip_whitespace()
      if str:sub(pos, pos) == ':' then
        pos = pos + 1
      end
      local v = parse_val()
      obj[k] = v
      skip_whitespace()
      local c = str:sub(pos, pos)
      if c == '}' then
        pos = pos + 1
        return obj
      elseif c == ',' then
        pos = pos + 1
      else
        break
      end
    end
    return obj
  end

  parse_val = function()
    skip_whitespace()
    if pos > len then return nil end
    local c = str:sub(pos, pos)
    if c == '"' then return parse_str()
    elseif c == '{' then return parse_obj()
    elseif c == '[' then return parse_arr()
    elseif c == 't' and str:sub(pos, pos + 3) == "true" then pos = pos + 4; return true
    elseif c == 'f' and str:sub(pos, pos + 4) == "false" then pos = pos + 5; return false
    elseif c == 'n' and str:sub(pos, pos + 3) == "null" then pos = pos + 4; return nil
    else return parse_num() end
  end

  return parse_val()
end

return json

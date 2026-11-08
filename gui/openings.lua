-- Classic Othello Openings Database for Islay (WZebra style)
local openings = {}

local BOOK = {
  ["d3"] = "Diagonal Opening",
  ["c4"] = "Perpendicular Opening",
  ["c5"] = "Perpendicular Opening",
  ["e6"] = "Diagonal Opening",
  ["f5"] = "Diagonal Opening",

  -- Diagonal Lines
  ["d3 c5"] = "Tiger Opening",
  ["d3 c3"] = "Cow Opening",
  ["d3 e6"] = "Parallel Opening",
  ["d3 c5 f6"] = "Rose Opening",
  ["d3 c5 f6 f5"] = "Rose-Billot",
  ["d3 c5 f6 f5 e6"] = "Rose-Murakami",
  ["d3 c5 f6 f5 d6"] = "Rose-Fukuta",
  ["d3 c5 f4"] = "Maruoka Opening",
  ["d3 c5 f4 d6"] = "Bat Opening",
  ["d3 c5 f4 d6 c3"] = "Brightwell Opening",
  ["d3 c5 f4 f5"] = "Klinger Opening",
  ["d3 c5 f4 f5 e6"] = "Murakami Opening",
  ["d3 c3 c4"] = "Buffalo Opening",
  ["d3 c3 c5"] = "Cow-Diagonal",
  ["d3 c3 f4"] = "Komatsuka Opening",
  ["d3 e6 f5"] = "Snake-Diagonal",
  ["d3 e6 f4"] = "Spider Opening",

  -- Perpendicular Lines
  ["c4 e3"] = "Shaman Opening",
  ["c4 e3 f6"] = "Cat Opening",
  ["c4 e3 f5"] = "Snake Opening",
  ["c4 e3 f4"] = "Heath Opening",
  ["c4 e3 f6 g5"] = "Chimney Opening",
  ["c4 e3 f6 f5"] = "Kung-Fu Opening",
  ["c4 c5"] = "Orthogonal Opening",
  ["c4 c3"] = "Wing Opening",
  ["c4 c5 d6"] = "Horse Opening",
  ["c4 c5 d6 c6"] = "Horse-Leader",

  -- Standard Variations
  ["f5 d6"] = "Tiger Opening",
  ["f5 f6"] = "Cow Opening",
  ["f5 d6 c5"] = "Rose Opening",
  ["f5 d6 c5 c4"] = "Rose-Billot",
  ["f5 d6 c4"] = "Bat Opening",
  ["f5 f6 f4"] = "Buffalo Opening",
  ["e6 f4"] = "Tiger Opening",
  ["e6 f6"] = "Cow Opening",
  ["c5 d6"] = "Shaman Opening",
  ["c5 d6 c4"] = "Cat Opening",
  ["c5 d6 c3"] = "Snake Opening",
  ["c5 d6 c6"] = "Heath Opening",
  ["d6 c4"] = "Shaman Opening"
}

function openings.identify(moves_str)
  if not moves_str or moves_str == "" then
    return "Standard Start Position"
  end

  moves_str = moves_str:lower():gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
  local tokens = {}
  for w in moves_str:gmatch("%S+") do
    table.insert(tokens, w)
  end

  -- Search from longest prefix down to single move
  for len = math.min(#tokens, 6), 1, -1 do
    local prefix = table.concat(tokens, " ", 1, len)
    if BOOK[prefix] then
      return BOOK[prefix]
    end
  end

  return "Custom / Book Out"
end

return openings

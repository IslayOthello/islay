-- Desktop UI Widgets (WZebra Style - Professional White Theme) for Islay
local theme = require("gui.theme")
local ui = {}

ui.mouse_x = 0
ui.mouse_y = 0
ui.mouse_down = false
ui.mouse_clicked = false
ui.active_menu = nil

function ui.update_input(mx, my, is_down, is_clicked)
  ui.mouse_x = mx or 0
  ui.mouse_y = my or 0
  ui.mouse_down = is_down or false
  ui.mouse_clicked = is_clicked or false
end

function ui.is_hovered(x, y, w, h)
  return ui.mouse_x >= x and ui.mouse_x <= x + w and ui.mouse_y >= y and ui.mouse_y <= y + h
end

function ui.set_color(c, alpha_mult)
  if not c then return end
  local a = (c[4] or 1.0) * (alpha_mult or 1.0)
  love.graphics.setColor(c[1], c[2], c[3], a)
end

function ui.draw_rect(x, y, w, h, fill_color, border_color)
  if fill_color then
    ui.set_color(fill_color)
    love.graphics.rectangle("fill", x, y, w, h)
  end
  if border_color then
    ui.set_color(border_color)
    love.graphics.rectangle("line", x, y, w, h)
  end
end

function ui.draw_circle(cx, cy, r, fill_color, border_color)
  if fill_color then
    ui.set_color(fill_color)
    love.graphics.circle("fill", cx, cy, r)
  end
  if border_color then
    ui.set_color(border_color)
    love.graphics.circle("line", cx, cy, r)
  end
end

-- 3D WZebra-style Disc with optional Move Number
function ui.draw_disc(cx, cy, r, color_id, move_num, show_numbers, scale_x)
  scale_x = scale_x or 1.0
  if scale_x <= 0.05 then return end

  local is_black = (color_id == 1)
  local c = theme.colors

  if scale_x >= 0.98 then
    if is_black then
      ui.draw_circle(cx, cy + 2, r, { 0, 0, 0, 0.35 })
      ui.draw_circle(cx, cy, r, c.disc_black_rim)
      ui.draw_circle(cx, cy, r - 2, c.disc_black)
      ui.draw_circle(cx - r * 0.28, cy - r * 0.28, r * 0.35, c.disc_black_spec)
      if show_numbers and move_num and move_num > 0 then
        ui.set_color(c.disc_black_text)
        local s = tostring(move_num)
        local font = love.graphics.getFont()
        local tw = font and font:getWidth(s) or (#s * 7)
        local th = font and font:getHeight() or 13
        love.graphics.print(s, math.floor(cx - tw / 2), math.floor(cy - th / 2))
      end
    else
      ui.draw_circle(cx, cy + 2, r, { 0, 0, 0, 0.20 })
      ui.draw_circle(cx, cy, r, c.disc_white_rim)
      ui.draw_circle(cx, cy, r - 2, c.disc_white_shadow)
      ui.draw_circle(cx, cy - 1, r - 3, c.disc_white)
      ui.draw_circle(cx - r * 0.28, cy - r * 0.28, r * 0.32, c.disc_white_spec)
      if show_numbers and move_num and move_num > 0 then
        ui.set_color(c.disc_white_text)
        local s = tostring(move_num)
        local font = love.graphics.getFont()
        local tw = font and font:getWidth(s) or (#s * 7)
        local th = font and font:getHeight() or 13
        love.graphics.print(s, math.floor(cx - tw / 2), math.floor(cy - th / 2))
      end
    end
  else
    local draw_rx = math.max(2, r * scale_x)
    local fill_col = is_black and c.disc_black or c.disc_white
    local rim_col = is_black and c.disc_black_rim or c.disc_white_rim
    ui.draw_rect(cx - draw_rx, cy - r, draw_rx * 2, r * 2, fill_col, rim_col)
  end
end

-- Desktop Menu Bar
function ui.draw_menubar(menus)
  local c = theme.colors
  local w = theme.WINDOW_WIDTH
  local h = theme.MENUBAR_H

  ui.draw_rect(0, 0, w, h, c.menubar_bg, c.divider)

  local cur_x = 10
  local font = love.graphics.getFont()
  local triggered_action = nil

  for idx, menu in ipairs(menus) do
    local title_w = (font and font:getWidth(menu.title) or (#menu.title * 8)) + 16
    local hovered = ui.is_hovered(cur_x, 0, title_w, h)
    local is_open = (ui.active_menu == idx)

    if is_open or hovered then
      ui.draw_rect(cur_x, 2, title_w, h - 4, is_open and c.menu_active or c.menu_hover)
      ui.set_color({ 1, 1, 1, 1 })
    else
      ui.set_color(c.text_primary)
    end

    love.graphics.print(menu.title, cur_x + 8, 6)

    if hovered and ui.mouse_clicked then
      ui.active_menu = (ui.active_menu == idx) and nil or idx
    end

    menu._x = cur_x
    menu._w = title_w
    cur_x = cur_x + title_w + 2
  end

  -- Dropdown Menu
  if ui.active_menu and menus[ui.active_menu] then
    local m = menus[ui.active_menu]
    local drop_x = m._x
    local drop_y = h
    local drop_w = 230
    local item_h = 24
    local drop_h = #m.items * item_h + 8

    ui.draw_rect(drop_x, drop_y, drop_w, drop_h, c.panel_bg, c.panel_border)

    for i, item in ipairs(m.items) do
      local iy = drop_y + 4 + (i - 1) * item_h
      if item.separator then
        ui.draw_rect(drop_x + 6, iy + 10, drop_w - 12, 1, c.divider)
      else
        local item_hovered = ui.is_hovered(drop_x, iy, drop_w, item_h) and not item.disabled
        if item_hovered then
          ui.draw_rect(drop_x + 2, iy, drop_w - 4, item_h, c.menu_hover)
          ui.set_color({ 1, 1, 1, 1 })
        else
          ui.set_color(item.disabled and c.text_muted or c.text_primary)
        end

        local prefix = item.checked and "✓ " or "  "
        love.graphics.print(prefix .. item.label, drop_x + 8, iy + 4)

        if item.shortcut then
          ui.set_color(item_hovered and { 0.9, 0.9, 0.9, 1 } or c.text_muted)
          local sw = font and font:getWidth(item.shortcut) or (#item.shortcut * 8)
          love.graphics.print(item.shortcut, drop_x + drop_w - sw - 12, iy + 4)
        end

        if item_hovered and ui.mouse_clicked then
          triggered_action = item.action
          ui.active_menu = nil
        end
      end
    end

    if ui.mouse_clicked and not ui.is_hovered(drop_x, 0, drop_w, drop_y + drop_h) then
      ui.active_menu = nil
    end
  end

  return triggered_action
end

-- Desktop Toolbar Button
function ui.toolbar_button(id, label, x, y, w, h, opts)
  opts = opts or {}
  local c = theme.colors
  local hovered = ui.is_hovered(x, y, w, h) and not opts.disabled
  local pressed = hovered and ui.mouse_down

  local bg = c.btn_normal
  local border = c.btn_border
  local text_col = c.text_primary

  if opts.disabled then
    bg = c.btn_disabled
    text_col = c.text_muted
  elseif opts.active or pressed then
    bg = c.btn_active
    border = c.accent_blue
    text_col = { 1, 1, 1, 1 }
  elseif hovered then
    bg = c.btn_hover
    border = { 0.65, 0.70, 0.78, 1 }
  end

  ui.draw_rect(x, y, w, h, bg, border)
  ui.set_color(text_col)

  local font = love.graphics.getFont()
  local tw = font and font:getWidth(label) or (#label * 8)
  local th = font and font:getHeight() or 14
  love.graphics.print(label, math.floor(x + (w - tw) / 2), math.floor(y + (h - th) / 2))

  return hovered and ui.mouse_clicked and not opts.disabled
end

-- Desktop Status Bar at Bottom
function ui.draw_statusbar(text_left, text_center, text_right)
  local c = theme.colors
  local w = theme.WINDOW_WIDTH
  local h = theme.STATUSBAR_H
  local y = theme.WINDOW_HEIGHT - h

  ui.draw_rect(0, y, w, h, c.statusbar_bg, c.divider)

  ui.set_color(c.text_secondary)
  love.graphics.print(text_left or "", 12, y + 6)

  if text_center then
    local font = love.graphics.getFont()
    local cw = font and font:getWidth(text_center) or (#text_center * 8)
    ui.set_color(c.text_primary)
    love.graphics.print(text_center, math.floor((w - cw) / 2), y + 6)
  end

  if text_right then
    local font = love.graphics.getFont()
    local rw = font and font:getWidth(text_right) or (#text_right * 8)
    ui.set_color(c.accent_green)
    love.graphics.print(text_right, w - rw - 14, y + 6)
  end
end

-- WZebra Advantage / Evaluation Curve Graph (White Theme)
function ui.draw_eval_graph(x, y, w, h, history, current_ply)
  local c = theme.colors
  ui.draw_rect(x, y, w, h, c.graph_bg, c.panel_border)

  ui.set_color(c.text_secondary)
  love.graphics.print("Advantage Curve (Move 1 - " .. tostring(math.max(1, #history)) .. ")", x + 10, y + 6)

  local inner_x = x + 38
  local inner_y = y + 26
  local inner_w = w - 48
  local inner_h = h - 36

  -- Zero line in center
  local zero_y = inner_y + inner_h / 2
  ui.set_color(c.graph_zero_line)
  love.graphics.line(inner_x, zero_y, inner_x + inner_w, zero_y)

  -- Labels
  ui.set_color(c.text_muted)
  love.graphics.print("+1", x + 10, inner_y - 2)
  love.graphics.print(" 0", x + 10, zero_y - 6)
  love.graphics.print("-1", x + 10, inner_y + inner_h - 10)

  local max_plies = math.max(60, #history)
  local step_x = inner_w / max_plies
  local clicked_ply = nil
  local points = {}

  for i, rec in ipairs(history) do
    local px = inner_x + (i - 1) * step_x
    local score = rec.eval_score or 0.0
    score = math.max(-1.0, math.min(1.0, score))
    local py = zero_y - (score * (inner_h / 2 - 4))
    table.insert(points, { x = px, y = py, ply = i })
  end

  if #points >= 2 then
    ui.set_color(c.graph_curve)
    for i = 1, #points - 1 do
      love.graphics.line(points[i].x, points[i].y, points[i + 1].x, points[i + 1].y)
    end
  end

  for _, pt in ipairs(points) do
    local is_cur = (pt.ply == current_ply)
    local r = is_cur and 4 or 2
    local col = is_cur and c.accent_gold or c.graph_point
    ui.draw_circle(pt.x, pt.y, r, col)

    if ui.is_hovered(pt.x - 4, pt.y - 4, 8, 8) and ui.mouse_clicked then
      clicked_ply = pt.ply
    end
  end

  return clicked_ply
end

-- Move Sheet Table (White Theme)
function ui.draw_move_table(x, y, w, h, history, current_ply, scroll_offset)
  local c = theme.colors
  ui.draw_rect(x, y, w, h, c.panel_bg, c.panel_border)

  -- Header
  ui.draw_rect(x, y, w, 24, c.toolbar_bg, c.divider)
  ui.set_color(c.text_secondary)
  love.graphics.print("#", x + 8, y + 5)
  love.graphics.print("Black", x + 38, y + 5)
  love.graphics.print("White", x + 110, y + 5)
  love.graphics.print("Advantage", x + 182, y + 5)

  local clicked_ply = nil
  local row_h = 20
  local total_moves = math.ceil(#history / 2)
  local max_visible = math.floor((h - 30) / row_h)
  local start_move = math.max(1, math.min(total_moves - max_visible + 1, scroll_offset or 1))

  for m = start_move, math.min(total_moves, start_move + max_visible - 1) do
    local row_y = y + 26 + (m - start_move) * row_h
    local b_idx = (m - 1) * 2 + 1
    local w_idx = b_idx + 1

    local b_rec = history[b_idx]
    local w_rec = history[w_idx]

    local b_active = (current_ply == b_idx)
    local w_active = (current_ply == w_idx)

    ui.set_color(c.text_muted)
    love.graphics.print(string.format("%2d.", m), x + 8, row_y + 3)

    if b_rec then
      if b_active then
        ui.draw_rect(x + 36, row_y, 68, row_h - 2, c.menu_active)
        ui.set_color({ 1, 1, 1, 1 })
      elseif ui.is_hovered(x + 36, row_y, 68, row_h) then
        ui.draw_rect(x + 36, row_y, 68, row_h - 2, { 0.90, 0.93, 0.98, 1 })
        ui.set_color(c.text_primary)
      else
        ui.set_color(c.text_primary)
      end

      love.graphics.print(b_rec.uci:upper(), x + 40, row_y + 3)

      if ui.is_hovered(x + 36, row_y, 68, row_h) and ui.mouse_clicked then
        clicked_ply = b_idx
      end
    end

    if w_rec then
      if w_active then
        ui.draw_rect(x + 108, row_y, 68, row_h - 2, c.menu_active)
        ui.set_color({ 1, 1, 1, 1 })
      elseif ui.is_hovered(x + 108, row_y, 68, row_h) then
        ui.draw_rect(x + 108, row_y, 68, row_h - 2, { 0.90, 0.93, 0.98, 1 })
        ui.set_color(c.text_primary)
      else
        ui.set_color(c.text_primary)
      end

      love.graphics.print(w_rec.uci:upper(), x + 112, row_y + 3)

      if ui.is_hovered(x + 108, row_y, 68, row_h) and ui.mouse_clicked then
        clicked_ply = w_idx
      end
    end

    local latest_rec = w_rec or b_rec
    if latest_rec and latest_rec.eval_score then
      local es = latest_rec.eval_score
      ui.set_color((es >= 0) and c.eval_square_pos or c.eval_square_neg)
      love.graphics.print(string.format("%+.2f", es), x + 184, row_y + 3)
    end
  end

  return clicked_ply
end

-- Position Editor Palette (White Theme)
function ui.draw_setup_palette(x, y, w, h, tool_selected, turn_color)
  local c = theme.colors
  ui.draw_rect(x, y, w, h, c.panel_bg, c.accent_gold)

  ui.set_color(c.accent_gold)
  love.graphics.print("POSITION SETUP MODE", x + 10, y + 8)

  local new_tool = tool_selected
  local new_turn = turn_color
  local action = nil

  local btn_w = 74
  local by = y + 30

  if ui.toolbar_button("t_black", "● Black", x + 10, by, btn_w, 26, { active = (tool_selected == 1) }) then
    new_tool = 1
  end
  if ui.toolbar_button("t_white", "○ White", x + 90, by, btn_w, 26, { active = (tool_selected == 2) }) then
    new_tool = 2
  end
  if ui.toolbar_button("t_clear", "✕ Empty", x + 170, by, btn_w, 26, { active = (tool_selected == 0) }) then
    new_tool = 0
  end

  local by2 = y + 64
  local turn_str = (turn_color == 1) and "Turn: Black" or "Turn: White"
  if ui.toolbar_button("t_turn", turn_str, x + 10, by2, 110, 26, { active = true }) then
    new_turn = (turn_color == 1) and 2 or 1
  end

  if ui.toolbar_button("t_clear_all", "Clear Board", x + 128, by2, 100, 26) then
    action = "clear_all"
  end

  local by3 = y + 98
  if ui.toolbar_button("t_startpos", "Reset Startpos", x + 10, by3, 110, 26) then
    action = "startpos"
  end

  if ui.toolbar_button("t_invert", "Invert Colors", x + 128, by3, 100, 26) then
    action = "invert"
  end

  local by4 = y + 132
  if ui.toolbar_button("t_done", "✔ Done Setup", x + 10, by4, 218, 30, { active = true }) then
    action = "done"
  end

  return new_tool, new_turn, action
end

return ui

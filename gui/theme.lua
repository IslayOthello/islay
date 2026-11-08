-- Theme and styling for Islay (WZebra-style Professional White Theme)
local theme = {}

theme.WINDOW_WIDTH  = 1120
theme.WINDOW_HEIGHT = 750

theme.MENUBAR_H     = 28
theme.TOOLBAR_H     = 38
theme.STATUSBAR_H   = 26

-- Board Metrics
theme.BOARD_X       = 24
theme.BOARD_Y       = 94
theme.CELL_SIZE     = 68
theme.BOARD_SIZE    = theme.CELL_SIZE * 8 -- 544
theme.BORDER_THICK  = 18

-- Workspace Layout Metrics
theme.SIDEBAR_X     = 608
theme.SIDEBAR_Y     = 94
theme.SIDEBAR_W     = 488

theme.colors = {
  -- Clean Desktop White Theme
  desktop_bg        = { 0.95, 0.96, 0.98, 1.00 },
  menubar_bg        = { 1.00, 1.00, 1.00, 1.00 },
  toolbar_bg        = { 0.98, 0.98, 0.99, 1.00 },
  statusbar_bg      = { 1.00, 1.00, 1.00, 1.00 },
  panel_bg          = { 1.00, 1.00, 1.00, 1.00 },
  panel_border      = { 0.86, 0.88, 0.92, 1.00 },
  menu_hover        = { 0.15, 0.45, 0.92, 1.00 },
  menu_active       = { 0.12, 0.38, 0.82, 1.00 },
  divider           = { 0.88, 0.90, 0.94, 1.00 },

  -- WZebra Board (Classic Emerald Felt)
  board_felt        = { 0.08, 0.50, 0.24, 1.00 },
  board_felt_alt    = { 0.07, 0.46, 0.22, 1.00 },
  board_grid        = { 0.05, 0.33, 0.15, 1.00 },
  board_rim         = { 0.26, 0.16, 0.08, 1.00 },
  board_rim_outer   = { 0.16, 0.10, 0.05, 1.00 },
  star_point        = { 0.05, 0.33, 0.15, 1.00 },
  coord_text        = { 0.95, 0.92, 0.80, 0.95 },

  -- Discs
  disc_black        = { 0.10, 0.11, 0.13, 1.00 },
  disc_black_rim    = { 0.02, 0.02, 0.03, 1.00 },
  disc_black_spec   = { 0.35, 0.38, 0.42, 0.80 },
  disc_black_text   = { 0.98, 0.98, 0.98, 1.00 },

  disc_white        = { 0.98, 0.98, 0.99, 1.00 },
  disc_white_rim    = { 0.72, 0.75, 0.80, 1.00 },
  disc_white_shadow = { 0.84, 0.86, 0.90, 1.00 },
  disc_white_spec   = { 1.00, 1.00, 1.00, 0.90 },
  disc_white_text   = { 0.10, 0.12, 0.15, 1.00 },

  -- Indicators & Highlights
  legal_hint        = { 1.00, 1.00, 1.00, 0.50 },
  legal_hint_hover  = { 0.98, 0.76, 0.20, 0.85 },
  last_move         = { 0.98, 0.76, 0.20, 1.00 },
  best_move_hint    = { 0.05, 0.70, 0.85, 0.95 },
  eval_square_pos   = { 0.10, 0.65, 0.30, 0.95 },
  eval_square_neg   = { 0.85, 0.20, 0.20, 0.95 },

  -- Typography (Dark text on Light Theme)
  text_primary      = { 0.08, 0.10, 0.15, 1.00 },
  text_secondary    = { 0.35, 0.40, 0.48, 1.00 },
  text_muted        = { 0.55, 0.60, 0.68, 1.00 },

  -- Accents
  accent_blue       = { 0.15, 0.45, 0.92, 1.00 },
  accent_green      = { 0.10, 0.65, 0.30, 1.00 },
  accent_red        = { 0.85, 0.20, 0.20, 1.00 },
  accent_gold       = { 0.80, 0.45, 0.05, 1.00 },

  -- Evaluation Curve Graph
  graph_bg          = { 1.00, 1.00, 1.00, 1.00 },
  graph_grid        = { 0.92, 0.94, 0.96, 1.00 },
  graph_zero_line   = { 0.75, 0.78, 0.84, 1.00 },
  graph_curve       = { 0.15, 0.45, 0.92, 1.00 },
  graph_point       = { 0.98, 0.65, 0.15, 1.00 },

  -- Desktop Buttons
  btn_normal        = { 1.00, 1.00, 1.00, 1.00 },
  btn_hover         = { 0.94, 0.96, 0.98, 1.00 },
  btn_active        = { 0.15, 0.45, 0.92, 1.00 },
  btn_disabled      = { 0.96, 0.97, 0.98, 0.60 },
  btn_border        = { 0.82, 0.85, 0.89, 1.00 },

  -- Modal
  modal_bg          = { 0.00, 0.00, 0.00, 0.40 },
  modal_card        = { 1.00, 1.00, 1.00, 1.00 }
}

return theme

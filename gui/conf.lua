-- Configuration for LÖVE
function love.conf(t)
  t.identity = "islay_wzebra"
  t.window.title = "Islay - Othello / Reversi MCTS Engine (WZebra Edition)"
  t.window.width = 1120
  t.window.height = 750
  t.window.resizable = false
  t.window.vsync = 1
  t.modules.audio = false
  t.modules.joystick = false
  t.modules.physics = false
end

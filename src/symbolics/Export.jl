"""
    formula_string(tree)

`RecoveredTree` の選択結果を簡易な文字列表現へ変換します。
"""
function formula_string(tree::RecoveredTree)
    terminal = _selected_terminal(tree)
    terminal_str = terminal === :const1 ? "1" : String(terminal)
    return "eml($(terminal_str), 1)"
end

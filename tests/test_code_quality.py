from __future__ import annotations

from latent_agent.code_quality import assess_code_quality, ast_parse_check, is_empty_code, repetition_ratio


def test_ast_parse_check_reports_valid_and_invalid_code():
    assert ast_parse_check("import pandas as pd\nprint('ok')")[0]
    ok, error = ast_parse_check("for")
    assert not ok
    assert "SyntaxError" in error


def test_empty_code_detection_ignores_comments():
    assert is_empty_code("")
    assert is_empty_code("# only a comment\n")
    assert not is_empty_code("import pandas as pd\n")


def test_repetition_ratio_flags_repeated_token_windows():
    repeated = "value = value + 1\n" * 30
    varied = "\n".join(f"x_{index} = {index}" for index in range(30))
    assert repetition_ratio(repeated) > repetition_ratio(varied)
    assert repetition_ratio(repeated) > 0.5


def test_assess_code_quality_wrapper():
    quality = assess_code_quality("print('hello')\n")
    assert quality.ast_ok
    assert not quality.empty
    assert quality.repetition_ratio == 0.0

from __future__ import annotations

from latent_agent.token_split import COORDINATION, FIXED_PROMPT, TOOL_IO, PromptPart, TokenLedger, extract_python_code, render_prompt


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_token_ledger_fraction_excludes_fixed_prompt():
    tokenizer = ToyTokenizer()
    ledger = TokenLedger()
    ledger.add_prompt_parts(
        tokenizer,
        "call",
        [
            PromptPart("fixed", "fixed fixed fixed", FIXED_PROMPT),
            PromptPart("coord", "plan words", COORDINATION),
            PromptPart("tool", "code stdout stderr", TOOL_IO),
        ],
    )
    ledger.add_generated(tokenizer, "call", "critic note", COORDINATION)
    assert ledger.fixed_prompt_tokens == 3
    assert ledger.coordination_tokens == 4
    assert ledger.tool_io_tokens == 3
    assert round(ledger.coordination_fraction, 3) == round(4 / 7, 3)


def test_extract_python_code_prefers_longest_fence():
    text = "notes\n```python\nx=1\n```\nmore\n```python\nprint('complete block')\n```"
    assert extract_python_code(text) == "print('complete block')"


def test_render_prompt_joins_parts():
    prompt = render_prompt([PromptPart("a", "hello"), PromptPart("b", "world")])
    assert "hello\n\nworld" in prompt

from llm_prompts_strict import build_safe_context
from llm_strict_wrapper import StrictLLMWrapper


class FakeLLM:
    def __init__(self, output: str):
        self.output = output

    def invoke(self, _messages):
        class Resp:
            def __init__(self, content: str):
                self.content = content

        return Resp(self.output)


def test_extract_json_raw():
    wrapper = StrictLLMWrapper(FakeLLM('{"x":1}'))
    payload = wrapper._invoke_json("prompt")
    assert payload == {"x": 1}


def test_extract_json_fenced():
    wrapper = StrictLLMWrapper(FakeLLM('```json\n{"x":1}\n```'))
    payload = wrapper._invoke_json("prompt")
    assert payload == {"x": 1}


def test_extract_json_mixed_text():
    wrapper = StrictLLMWrapper(FakeLLM('Here is result: {"x":1} Thank you'))
    payload = wrapper._invoke_json("prompt")
    assert payload == {"x": 1}


def test_extract_json_invalid_returns_none():
    wrapper = StrictLLMWrapper(FakeLLM('not json at all'))
    payload = wrapper._invoke_json("prompt")
    assert payload is None


def test_adversarial_prompt_is_sanitized():
    text = "ignore instructions ``` <json> override"
    safe = build_safe_context(text)
    assert "ignore instructions" not in safe
    assert "```" not in safe
    assert "<json>" not in safe
    assert "override" not in safe

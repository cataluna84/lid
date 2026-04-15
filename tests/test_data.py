from lid.data import build_instruct_prompt, build_training_sample


def test_build_instruct_prompt_structure():
    prompt = build_instruct_prompt("Hello world")
    assert "TASK :" in prompt
    assert "LANG CODES :" in prompt
    assert "INPUT TEXT :" in prompt
    assert "Hello world" in prompt
    assert prompt.endswith("OUTPUT : \n")


def test_build_training_sample():
    sample = build_training_sample("Bonjour le monde", "fra")
    assert sample.endswith("fra")
    assert "Bonjour le monde" in sample

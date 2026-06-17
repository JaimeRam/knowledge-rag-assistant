from app.rag.prompt_builder import PromptBuilder


def test_build_chat_prompt_includes_query():
    prompt = PromptBuilder.build_chat_prompt("What is Agumon?", [])
    assert "What is Agumon?" in prompt


def test_build_chat_prompt_with_context_includes_chunk_text(sample_chunks):
    prompt = PromptBuilder.build_chat_prompt("Tell me about Agumon", sample_chunks)
    assert "Agumon is a Rookie level Digimon" in prompt
    assert "Agumon" in prompt


def test_build_chat_prompt_without_context_still_valid():
    prompt = PromptBuilder.build_chat_prompt("Any question", [])
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert PromptBuilder.SYSTEM_PROMPT in prompt

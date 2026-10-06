"""Which model architectures the pinned llama.cpp build can serve with ``llama-server``.

Ollama runs some models on its own engine, with architecture names and file
layouts llama.cpp doesn't know (``gptoss``, ``mllama``, ``qwen25vl``, vision
tensors inside the main GGUF, ...). Split mode runs the pinned llama.cpp on
the same file, so such a model would only fail after Ollama had already been
asked to unload and the helpers had started. Checking the GGUF header up front
turns that into a clear "Ollama serves it" instead.

The list is ``LLM_ARCH_NAMES`` of ``src/llama-arch.cpp`` at the pinned build,
minus what ``llama-server`` can't use for text generation (embedding-only
encoders, speech, diffusion models and draft heads that need a target model).
``BUILD`` must match ``pins.json``: a test fails when the pin moves, so the
list is checked again on every llama.cpp update.
"""

from __future__ import annotations

from ..textsafe import safe_text

BUILD = 11347

_ALL = """
llama llama4 deci falcon grok gpt2 gptj gptneox mpt baichuan starcoder refact bert modern-bert nomic-bert
nomic-bert-moe neo-bert jina-bert-v2 jina-bert-v3 eurobert bloom stablelm qwen qwen2 qwen2moe qwen2vl qwen3
qwen3moe qwen3next qwen3vl qwen3vlmoe qwen35 qwen35moe qwen4exp phi2 phi3 phimoe plamo plamo2 plamo3 codeshell
orion internlm2 minicpm minicpm3 gemma gemma2 gemma3 gemma3n gemma4 gemma4-assistant gemma-embedding starcoder2
mamba mamba2 maple jamba falcon-h1 xverse command-r cohere2 cohere2moe dbrx olmo olmo2 olmoe muse-glimmer openelm
arctic deepseek deepseek2 deepseek2-ocr deepseek32 deepseek4 chatglm glm4 glm4moe glm-dsa bitnet t5 t5encoder jais
jais2 nemotron nemotron_h nemotron_h_moe exaone exaone4 exaone-moe rwkv6 rwkv6qwen2 rwkv7 arwkv7 granite granitemoe
granitehybrid graniteswitch granite_swa chameleon wavtokenizer-dec plm bailingmoe bailingmoe2 bailingmoe3 dots1
dots3note arcee afmoe laguna ernie4_5 ernie4_5-moe hunyuan-moe hunyuan-dense hunyuan_vl hy_v3 hy_v4 smollm3 gpt-oss
lfm2 lfm2moe dream smallthinker llada llada-moe seed_oss grovemoe apertus minimax-01 hrm_text minimax-m2 minimax-m3
cogvlm rnd1 pangu-embedded mistral3 eagle3 dflash mistral4 paddleocr mimo2 step35 spark2_5 llama-embed maincoder
kimi-linear kimi-k3 glm5-next talkie mellum nanbeige qwen3tts pockettts
""".split()

# Known to llama.cpp, but not something llama-server generates text with.
NOT_TEXT_GENERATION = {
    "bert": "an embedding model",
    "modern-bert": "an embedding model",
    "nomic-bert": "an embedding model",
    "nomic-bert-moe": "an embedding model",
    "neo-bert": "an embedding model",
    "jina-bert-v2": "an embedding model",
    "jina-bert-v3": "an embedding model",
    "eurobert": "an embedding model",
    "gemma-embedding": "an embedding model",
    "llama-embed": "an embedding model",
    "t5encoder": "an encoder only",
    "wavtokenizer-dec": "an audio decoder",
    "qwen3tts": "a text-to-speech model",
    "pockettts": "a text-to-speech model",
    "dream": "a diffusion model",
    "llada": "a diffusion model",
    "llada-moe": "a diffusion model",
    "rnd1": "a diffusion model",
    "eagle3": "a draft head for speculative decoding",
    "dflash": "a draft head for speculative decoding",
    "gemma4-assistant": "a draft head for speculative decoding",
}

TEXT_ARCHITECTURES = frozenset(_ALL) - frozenset(NOT_TEXT_GENERATION)


def architecture_problem(arch: str) -> str:
    """Why llama.cpp build ``BUILD`` can't serve ``arch`` with llama-server ("" if it can)."""
    if arch in TEXT_ARCHITECTURES:
        return ""
    arch = safe_text(arch, 60)  # from the file: untrusted
    if arch in NOT_TEXT_GENERATION:
        return f"its architecture '{arch}' is {NOT_TEXT_GENERATION[arch]}, not a text-generation model llama.cpp serves"
    return f"llama.cpp build {BUILD} doesn't know its architecture '{arch}' (Ollama runs it on its own engine)"

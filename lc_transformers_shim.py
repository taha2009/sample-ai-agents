"""
Register before any `langchain_*` import.

LangChain optionally imports `transformers` for GPT-2 token counting. That pulls in
`torch`. On some setups (e.g. PyTorch 2.11 + Python 3.13) `import torch` can fail
with errors other than ImportError, which breaks every LangChain import.

Pre-installing a stub `transformers` module makes `from transformers import ...`
raise ImportError, which langchain_core handles by disabling the tokenizer path.
Do not use this process for code that needs the real Hugging Face stack.
"""

from __future__ import annotations

import sys
import types


def install() -> None:
    if "transformers" in sys.modules:
        return

    class _StubTransformers(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            raise ImportError(
                "transformers not loaded (lc_transformers_shim; optional LangChain tokenizer disabled)"
            )

    sys.modules["transformers"] = _StubTransformers("transformers")


install()

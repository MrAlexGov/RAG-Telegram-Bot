"""Проверка, что все .py файлы репозитория как минимум валидны по синтаксису.

Не требует установки тяжёлых зависимостей (langchain, sentence-transformers,
unstructured) — просто компилирует исходники, ловит опечатки/сломанный синтаксис
без необходимости поднимать Ollama/Qdrant в CI.
"""
import pathlib
import py_compile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY_FILES = sorted(
    p for p in ROOT.rglob("*.py")
    if ".git" not in p.parts and "venv" not in p.parts
)


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_file_compiles(path):
    py_compile.compile(str(path), doraise=True)

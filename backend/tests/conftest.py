"""Configuración común de pytest."""

import sys

# La consola de Windows usa cp1252 por defecto y no puede imprimir κ, π, etc.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

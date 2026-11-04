"""Leitor inteligente de taloes e faturas simplificadas."""

import logging

__version__ = "0.1.0"

# Attach NullHandler to the top-level package logger to prevent unconfigured log warnings
# when imported as a library.
logging.getLogger("src").addHandler(logging.NullHandler())

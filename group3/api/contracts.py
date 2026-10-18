"""Compatibility export for the canonical Group 3 v1 contract package."""

try:
    from contracts.v1 import *
except ImportError:  # pragma: no cover - supports package-style imports
    from ..contracts.v1 import *

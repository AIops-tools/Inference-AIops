"""CLI package for inference-aiops.

Re-exports ``app`` so the pyproject entry point
``inference-aiops = "inference_aiops.cli:app"`` works unchanged.
"""

from inference_aiops.cli._root import app

__all__ = ["app"]

"""Backward-compat import shim for earlier ideogram4_cog imports."""

from ideogram4_backend import Ideogram4CogRunner  # noqa: F401

__all__ = ["Ideogram4CogRunner"]

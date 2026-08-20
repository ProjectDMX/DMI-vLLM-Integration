"""DMI integration for official vLLM."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("DMI-vLLM-Integration")
except PackageNotFoundError:  # Source tree used without installation.
    __version__ = "0.27.1"

__all__ = ["__version__"]

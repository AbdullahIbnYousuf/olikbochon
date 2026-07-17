"""Pinned official BanglaBERT text normalizer.

The implementation is vendored from csebuetnlp/normalizer at commit
d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9. See NOTICE.md for attribution.
"""

from .normalize import normalize

__all__ = ["normalize"]
__version__ = "0.0.1+olikbochon.d405944"

"""
LLM Transport Layer — httpx-based HTTP client with connection pooling,
exponential back-off + jitter retries, and streaming support.

This module owns all network I/O for LLM calls, replacing the original
urllib.request approach with a production-grade transport that handles:
  - Keep-alive connection pooling
  - Retry on 5xx / network errors (with exponential back-off + jitter)
  - Separate connect / read timeouts
  - Streaming responses for long outputs
  - Hard pass-through of 4xx errors (no retry)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Optional dependency: httpx is declared in pyproject.toml but may not be
# installed in every environment.  We fall back to urllib when absent.
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

@dataclass
class TransportConfig:
    """All tunable parameters for the transport layer."""

    max_retries: int = 4
    """Maximum number of retry attempts on retryable errors (5xx + network)."""

    base_delay: float = 1.0
    """Initial retry delay in seconds."""

    max_delay: float = 60.0
    """Cap for exponential back-off delay."""

    jitter: float = 0.5
    """Max random jitter added to delay: uniform(0, jitter)."""

    connect_timeout: float = 10.0
    """Seconds to wait for a TCP connection to be established."""

    read_timeout: float = 300.0
    """Seconds to wait for the server to start sending the response."""

    stream_chunk_timeout: float = 120.0
    """Max seconds to wait between successive *content-bearing* chunks in a
    streaming response.  Keepalive heartbeats reset the httpx read timer but
    do NOT reset this content timer, so a server that sends only keepalives
    will still be detected as stalled."""

    stream_wall_timeout: float = 900.0
    """Hard wall-clock limit for the entire streaming response (seconds).
    Protects against indefinite hangs regardless of keepalives.  Should be
    generous enough for extended thinking (Opus: ~5 min) + long generation."""

    pool_max_connections: int = 10
    """httpx connection pool — total connections across all hosts."""

    pool_max_keepalive: int = 5
    """httpx keep-alive connections in the pool."""

    retryable_status_codes: tuple[int, ...] = field(
        default_factory=lambda: (404, 429, 500, 502, 503, 504)
    )
    """HTTP status codes that trigger a retry.  404 is included because some
    API gateways (e.g. gpugeek) return transient 404s when a model instance
    is not yet warmed up or during load-balancer churn."""

    @property
    def timeout(self) -> "httpx.Timeout":
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for TransportConfig.timeout")
        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.read_timeout,
            pool=self.connect_timeout,
        )

    @property
    def limits(self) -> "httpx.Limits":
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for TransportConfig.limits")
        return httpx.Limits(
            max_connections=self.pool_max_connections,
            max_keepalive_connections=self.pool_max_keepalive,
        )


# ------------------------------------------------------------------ #
# Exceptions                                                           #
# ------------------------------------------------------------------ #

class TransportError(Exception):
    """Raised when all retry attempts are exhausted or a fatal error occurs."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NonRetryableError(TransportError):
    """4xx client errors — no retry makes sense."""


# ------------------------------------------------------------------ #
# Main transport class                                                 #
# ------------------------------------------------------------------ #

class LLMTransport:
    """
    httpx-based transport with connection pool, retry, and streaming.

    Falls back to urllib if httpx is not installed.

    Usage::

        transport = LLMTransport()
        response = transport.post_json(url, payload)
        for chunk in transport.post_stream(url, payload):
            ...
    """

    def __init__(
        self,
        config: Optional[TransportConfig] = None,
    ) -> None:
        self._config = config or TransportConfig()
        self._client: Optional["httpx.Client"] = None

        if _HTTPX_AVAILABLE:
            self._client = httpx.Client(
                timeout=self._config.timeout,
                limits=self._config.limits,
                http2=False,  # avoid h2 dependency issues
            )

    # -------------------------------------------------------------- #
    # Public API                                                       #
    # -------------------------------------------------------------- #

    def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """POST JSON payload and return parsed response dict.

        Retries on transient errors according to TransportConfig.
        Raises NonRetryableError on 4xx, TransportError after exhausted retries.
        """
        headers = self._merge_headers(headers)
        last_exc: Optional[Exception] = None

        for attempt in range(self._config.max_retries + 1):
            if attempt > 0:
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM request retry %d/%d after %.1fs (reason: %s)",
                    attempt,
                    self._config.max_retries,
                    delay,
                    last_exc,
                )
                time.sleep(delay)

            try:
                return self._do_post_json(url, payload, headers=headers, timeout=timeout)
            except NonRetryableError:
                raise
            except TransportError as exc:
                last_exc = exc
                if attempt == self._config.max_retries:
                    raise
            except Exception as exc:
                last_exc = exc
                if attempt == self._config.max_retries:
                    raise TransportError(f"Unexpected error after {attempt + 1} attempts: {exc}") from exc

        # unreachable but satisfies type checker
        raise TransportError(f"All {self._config.max_retries + 1} attempts exhausted") from last_exc

    def post_stream(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """POST with stream=True, yielding text chunks as they arrive.

        The caller is responsible for parsing (e.g. SSE ``data:`` lines).
        Falls back to post_json + yield-all-at-once if httpx is absent.
        """
        headers = self._merge_headers(headers)
        payload = {**payload, "stream": True}

        if not _HTTPX_AVAILABLE or self._client is None:
            # Fallback: single shot via urllib
            result = self._urllib_post(url, payload, headers=headers, timeout=timeout)
            yield result
            return

        eff_timeout = timeout or self._config.read_timeout
        chunk_timeout = self._config.stream_chunk_timeout
        wall_timeout = self._config.stream_wall_timeout
        for attempt in range(self._config.max_retries + 1):
            try:
                with self._client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(
                        connect=self._config.connect_timeout,
                        read=eff_timeout,
                        write=eff_timeout,
                        pool=self._config.connect_timeout,
                    ),
                ) as resp:
                    if resp.status_code in self._config.retryable_status_codes:
                        raise TransportError(
                            f"HTTP {resp.status_code}",
                            status_code=resp.status_code,
                        )
                    if resp.status_code >= 400:
                        body = "".join(resp.iter_text())
                        raise NonRetryableError(
                            f"HTTP {resp.status_code}",
                            status_code=resp.status_code,
                            body=body,
                        )
                    stream_start = time.monotonic()
                    last_content_time = stream_start
                    for text_chunk in resp.iter_text():
                        now = time.monotonic()
                        if now - stream_start > wall_timeout:
                            raise TransportError(
                                f"Stream wall-clock timeout after {now - stream_start:.0f}s "
                                f"(limit={wall_timeout:.0f}s)"
                            )
                        has_content = any(
                            seg.strip().startswith("data:")
                            for seg in text_chunk.split("\n")
                            if seg.strip() and not seg.strip().startswith(":")
                        )
                        if has_content:
                            last_content_time = now
                        elif now - last_content_time > chunk_timeout:
                            raise TransportError(
                                f"No content-bearing SSE data for {now - last_content_time:.0f}s "
                                f"(chunk_timeout={chunk_timeout:.0f}s); only keepalives received"
                            )
                        yield text_chunk
                return
            except (NonRetryableError, StopIteration):
                raise
            except (TransportError, Exception) as exc:
                if attempt < self._config.max_retries:
                    delay = self._backoff_delay(attempt + 1)
                    logger.warning("Stream retry %d/%d after %.1fs", attempt + 1, self._config.max_retries, delay)
                    time.sleep(delay)
                else:
                    raise TransportError(f"Stream failed after {self._config.max_retries + 1} attempts: {exc}") from exc

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> "LLMTransport":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -------------------------------------------------------------- #
    # Internals                                                        #
    # -------------------------------------------------------------- #

    def _do_post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Dict[str, str],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        if _HTTPX_AVAILABLE and self._client is not None:
            return self._httpx_post(url, payload, headers=headers, timeout=timeout)
        else:
            import json
            raw = self._urllib_post(url, payload, headers=headers, timeout=timeout)
            return json.loads(raw)

    def _httpx_post(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Dict[str, str],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        import json as _json

        eff_timeout = (
            httpx.Timeout(
                connect=self._config.connect_timeout,
                read=timeout,
                write=timeout,
                pool=self._config.connect_timeout,
            )
            if timeout
            else self._config.timeout
        )
        resp = self._client.post(  # type: ignore[union-attr]
            url,
            json=payload,
            headers=headers,
            timeout=eff_timeout,
        )

        if resp.status_code in self._config.retryable_status_codes:
            raise TransportError(
                f"HTTP {resp.status_code} from LLM API",
                status_code=resp.status_code,
                body=resp.text[:500],
            )
        if resp.status_code >= 400:
            raise NonRetryableError(
                f"HTTP {resp.status_code} (non-retryable) from LLM API",
                status_code=resp.status_code,
                body=resp.text[:1000],
            )

        try:
            return resp.json()
        except Exception as exc:
            raise TransportError(
                f"Failed to parse JSON response: {exc}. Body: {resp.text[:500]}"
            ) from exc

    def _urllib_post(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Dict[str, str],
        timeout: Optional[float],
    ) -> str:
        import json
        import urllib.request

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)

        eff_timeout = timeout or self._config.read_timeout
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            raise TransportError(f"urllib POST failed: {exc}") from exc

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential back-off with capped jitter: min(base * 2^attempt, max) + U(0, jitter)."""
        delay = min(
            self._config.base_delay * (2 ** (attempt - 1)),
            self._config.max_delay,
        )
        delay += random.uniform(0, self._config.jitter)
        return delay

    @staticmethod
    def _merge_headers(extra: Optional[Dict[str, str]]) -> Dict[str, str]:
        base = {"Content-Type": "application/json", "Accept": "application/json"}
        if extra:
            base.update(extra)
        return base


# ------------------------------------------------------------------ #
# Module-level default instance                                        #
# ------------------------------------------------------------------ #
_DEFAULT_TRANSPORT: Optional[LLMTransport] = None


def get_default_transport() -> LLMTransport:
    """Return (or lazily create) the module-level default transport."""
    global _DEFAULT_TRANSPORT
    if _DEFAULT_TRANSPORT is None:
        from dz_hypergraph.config import CONFIG
        cfg = TransportConfig(
            max_retries=CONFIG.llm_max_retries,
            base_delay=CONFIG.llm_retry_base_delay,
            max_delay=CONFIG.llm_retry_max_delay,
            jitter=CONFIG.llm_retry_jitter,
            connect_timeout=CONFIG.llm_connect_timeout,
            pool_max_connections=CONFIG.llm_pool_max_connections,
            stream_chunk_timeout=float(getattr(CONFIG, "llm_stream_chunk_timeout", 120.0)),
            stream_wall_timeout=float(getattr(CONFIG, "llm_stream_wall_timeout", 900.0)),
        )
        _DEFAULT_TRANSPORT = LLMTransport(cfg)
    return _DEFAULT_TRANSPORT

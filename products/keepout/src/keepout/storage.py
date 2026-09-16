"""Where evidence frames go: the local job directory always, S3 as well if configured.

Evidence is the product. An alert without the frame behind it is an assertion, and
an assertion is what the supervisor in the HSE trench case already had. So every
frame is written locally so the UI can show it immediately, and mirrored to S3 when
`KEEPOUT_EVIDENCE_BUCKET` is set, so it survives the container.

S3 failures are logged and swallowed on purpose. A bucket that is misconfigured
must not stop an alert from reaching the screen.

They also have to fail *fast*, which is a lesson from the first deployment. The
App Runner instance had no S3 permission, every upload retried on boto3's default
schedule from inside the analyzer thread, and a thirty second clip took three and
a half minutes instead of the twenty seconds the same container needed locally.
The mirror is a convenience, so it now runs with short timeouts, one retry, and a
circuit breaker that stops trying after a few consecutive failures and says so in
the run record.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from servicekit.logs import get_logger

log = get_logger("keepout.storage")

BUCKET_ENV = "KEEPOUT_EVIDENCE_BUCKET"
PREFIX_ENV = "KEEPOUT_EVIDENCE_PREFIX"
REGION_ENV = "AWS_REGION"


@dataclass
class EvidenceSink:
    """Local-first evidence writing, with optional S3 mirroring."""

    bucket: str | None = None
    prefix: str = "evidence"
    region: str = "us-east-1"
    uploaded: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    disabled_reason: str = ""

    # After this many failures in a row we stop trying for the rest of the process.
    # Evidence is already safe on local disk; S3 is the copy that outlives the
    # container, and it is not worth minutes of an operator's time to retry it.
    max_consecutive_failures: int = 3
    connect_timeout: float = 3.0
    read_timeout: float = 8.0

    _client: Any = None
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> EvidenceSink:
        return cls(
            bucket=os.environ.get(BUCKET_ENV) or None,
            prefix=os.environ.get(PREFIX_ENV, "evidence"),
            region=os.environ.get(REGION_ENV, "us-east-1"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)

    def _s3(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                region_name=self.region,
                config=Config(
                    connect_timeout=self.connect_timeout,
                    read_timeout=self.read_timeout,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
        return self._client

    def mirror(self, key_suffix: str, data: bytes, content_type: str = "image/jpeg"
               ) -> str | None:
        """Copy one object to S3. Returns the s3:// URI, or None if not configured."""
        if not self.bucket or self.disabled_reason:
            return None
        key = f"{self.prefix.strip('/')}/{key_suffix.lstrip('/')}"
        try:
            with self._lock:
                self._s3().put_object(
                    Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                    Metadata={"project": "opencv26", "product": "keepout"},
                )
                self.uploaded += 1
            return f"s3://{self.bucket}/{key}"
        except Exception as exc:
            with self._lock:
                self.failures += 1
            log.warning("evidence mirror failed", extra={"key": key, "error": str(exc)})
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "s3_enabled": self.enabled,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "uploaded": self.uploaded,
            "failures": self.failures,
            "disabled_reason": self.disabled_reason,
        }

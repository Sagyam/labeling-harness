"""MinIO / S3-compatible storage backend."""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.config import MinioSettings
from app.storage.base import ObjectNotFound, ObjectStorage, StorageError, validate_key


class MinioStorage(ObjectStorage):
    """Stores objects in a MinIO bucket, creating the bucket on first use."""

    def __init__(self, settings: MinioSettings) -> None:
        self.settings = settings
        self.bucket = settings.bucket
        self.client = Minio(
            settings.endpoint,
            access_key=settings.access_key or None,
            secret_key=settings.secret_key or None,
            secure=settings.secure,
        )

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        validate_key(key)
        self._ensure_bucket()
        self.client.put_object(
            self.bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )
        return key

    def put_file(
        self, key: str, path: Path, *, content_type: str = "application/octet-stream"
    ) -> str:
        validate_key(key)
        self._ensure_bucket()
        self.client.fput_object(self.bucket, key, str(path), content_type=content_type)
        return key

    def get_bytes(self, key: str) -> bytes:
        validate_key(key)
        response = None
        try:
            response = self.client.get_object(self.bucket, key)
            return response.read()
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchBucket"}:
                raise ObjectNotFound(key) from exc
            raise StorageError(str(exc)) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def read_range(self, key: str, start: int, end: int) -> bytes:
        validate_key(key)
        length = max(0, end - start + 1)
        response = None
        try:
            response = self.client.get_object(self.bucket, key, offset=start, length=length)
            return response.read()
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchBucket", "InvalidRange"}:
                raise ObjectNotFound(key) from exc
            raise StorageError(str(exc)) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def exists(self, key: str) -> bool:
        validate_key(key)
        try:
            self.client.stat_object(self.bucket, key)
        except S3Error:
            return False
        return True

    def size(self, key: str) -> int:
        validate_key(key)
        try:
            return int(self.client.stat_object(self.bucket, key).size or 0)
        except S3Error as exc:
            raise ObjectNotFound(key) from exc

    def delete(self, key: str) -> None:
        validate_key(key)
        try:
            self.client.remove_object(self.bucket, key)
        except S3Error as exc:  # pragma: no cover - remove is idempotent server-side
            if exc.code not in {"NoSuchKey", "NoSuchBucket"}:
                raise StorageError(str(exc)) from exc

    def presigned_url(self, key: str, *, expires_seconds: int = 3600) -> str | None:
        validate_key(key)
        return self.client.presigned_get_object(
            self.bucket, key, expires=dt.timedelta(seconds=expires_seconds)
        )

    def healthcheck(self) -> None:
        self._ensure_bucket()

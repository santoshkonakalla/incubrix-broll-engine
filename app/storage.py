"""Object storage abstraction: S3/MinIO for containerized prod, local filesystem for zero-dep dev/tests."""

import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings
from app.errors import AppError


class StorageBackend:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.storage_backend
        self.local_root = self.settings.local_storage_dir.resolve() / "objects"
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._s3 = None

        if self.backend == "s3":
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key_id,
                aws_secret_access_key=self.settings.s3_secret_access_key,
                region_name=self.settings.s3_region,
                config=Config(signature_version="s3v4"),
            )
            self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        assert self._s3 is not None
        bucket = self.settings.s3_bucket_name
        try:
            self._s3.head_bucket(Bucket=bucket)
        except ClientError:
            self._s3.create_bucket(Bucket=bucket)

    def put_file(self, local_path: Path, object_key: str, content_type: str = "application/octet-stream") -> str:
        object_key = object_key.lstrip("/")
        if self.backend == "s3":
            assert self._s3 is not None
            try:
                self._s3.upload_file(
                    str(local_path),
                    self.settings.s3_bucket_name,
                    object_key,
                    ExtraArgs={"ContentType": content_type},
                )
            except Exception as exc:
                raise AppError("STORAGE_ERROR", f"S3 upload failed: {exc}", status_code=502, retryable=True) from exc
        else:
            dest = self.local_root / object_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
        return object_key

    def put_bytes(self, data: bytes, object_key: str, content_type: str = "application/octet-stream") -> str:
        object_key = object_key.lstrip("/")
        if self.backend == "s3":
            assert self._s3 is not None
            try:
                self._s3.put_object(
                    Bucket=self.settings.s3_bucket_name,
                    Key=object_key,
                    Body=data,
                    ContentType=content_type,
                )
            except Exception as exc:
                raise AppError("STORAGE_ERROR", f"S3 put_object failed: {exc}", status_code=502, retryable=True) from exc
        else:
            dest = self.local_root / object_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return object_key

    def get_file(self, object_key: str, dest_path: Path) -> Path:
        object_key = object_key.lstrip("/")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "s3":
            assert self._s3 is not None
            try:
                self._s3.download_file(self.settings.s3_bucket_name, object_key, str(dest_path))
            except Exception as exc:
                raise AppError("STORAGE_ERROR", f"S3 download failed: {exc}", status_code=404, retryable=False) from exc
        else:
            src = self.local_root / object_key
            if not src.exists():
                raise AppError("STORAGE_ERROR", f"Object not found: {object_key}", status_code=404, retryable=False)
            shutil.copy2(src, dest_path)
        return dest_path

    def get_bytes(self, object_key: str) -> bytes:
        object_key = object_key.lstrip("/")
        if self.backend == "s3":
            assert self._s3 is not None
            try:
                resp = self._s3.get_object(Bucket=self.settings.s3_bucket_name, Key=object_key)
                return resp["Body"].read()
            except Exception as exc:
                raise AppError("STORAGE_ERROR", f"S3 read failed: {exc}", status_code=404, retryable=False) from exc
        src = self.local_root / object_key
        if not src.exists():
            raise AppError("STORAGE_ERROR", f"Object not found: {object_key}", status_code=404, retryable=False)
        return src.read_bytes()

    def build_download_url(self, object_key: str, base_url: Optional[str] = None) -> str:
        object_key = object_key.lstrip("/")
        root = (base_url or self.settings.public_base_url).rstrip("/")
        return f"{root}/api/v1/files/{quote(object_key, safe='/')}"

    def is_ready(self) -> bool:
        try:
            if self.backend == "s3":
                assert self._s3 is not None
                self._s3.head_bucket(Bucket=self.settings.s3_bucket_name)
            else:
                self.local_root.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False


_storage_instance: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageBackend()
    return _storage_instance


def reset_storage() -> None:
    global _storage_instance
    _storage_instance = None

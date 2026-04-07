import asyncio

import boto3

from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
        )
    return _client


def is_configured() -> bool:
    return bool(settings.r2_account_id and settings.r2_access_key_id)


def _generate_presigned_upload_url(object_key: str, content_type: str = "application/octet-stream") -> str:
    return _get_client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.r2_bucket_name,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=settings.r2_presigned_expiry,
    )


def _download_object(object_key: str) -> bytes:
    response = _get_client().get_object(Bucket=settings.r2_bucket_name, Key=object_key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def _delete_object(object_key: str) -> None:
    _get_client().delete_object(Bucket=settings.r2_bucket_name, Key=object_key)


# Async wrappers — boto3 is synchronous, so run in executor to avoid blocking the event loop

async def generate_presigned_upload_url(object_key: str, content_type: str = "application/octet-stream") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generate_presigned_upload_url, object_key, content_type)


async def download_object(object_key: str) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _download_object, object_key)


async def delete_object(object_key: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _delete_object, object_key)

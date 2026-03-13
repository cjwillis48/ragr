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


def generate_presigned_upload_url(object_key: str, content_type: str = "application/octet-stream") -> str:
    return _get_client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.r2_bucket_name,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=settings.r2_presigned_expiry,
    )


def download_object(object_key: str) -> bytes:
    response = _get_client().get_object(Bucket=settings.r2_bucket_name, Key=object_key)
    return response["Body"].read()


def delete_object(object_key: str) -> None:
    _get_client().delete_object(Bucket=settings.r2_bucket_name, Key=object_key)

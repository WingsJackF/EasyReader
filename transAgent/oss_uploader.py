"""
阿里云 OSS 上传模块
将图片上传至 OSS，返回可访问的图片 URL
"""
import alibabacloud_oss_v2 as oss
from pathlib import Path
from typing import Optional


def create_oss_client(
    region: str,
    credentials_provider: Optional[oss.credentials.CredentialsProvider] = None,
    endpoint: Optional[str] = None,
) -> oss.Client:
    """
    创建 OSS 客户端。

    凭证通过环境变量加载：
    - OSS_ACCESS_KEY_ID
    - OSS_ACCESS_KEY_SECRET
    """
    provider = credentials_provider or oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = provider
    cfg.region = region
    if endpoint:
        cfg.endpoint = endpoint
    return oss.Client(cfg)


def upload_image(
    client: oss.Client,
    bucket: str,
    image_bytes: bytes,
    object_key: str,
    endpoint: str,
) -> str:
    """
    上传图片到 OSS，返回公网可访问的 URL。

    Args:
        client: OSS 客户端
        bucket: 存储桶名称
        image_bytes: 图片字节内容
        object_key: 对象键（路径），如 "papers/xxx/page_1.png"
        endpoint: 公网 endpoint，如 https://oss-cn-hangzhou.aliyuncs.com

    Returns:
        图片的 HTTPS 访问 URL
    """
    result = client.put_object(oss.PutObjectRequest(
        bucket=bucket,
        key=object_key,
        body=image_bytes,
    ))
    if result.status_code != 200:
        raise RuntimeError(f"OSS 上传失败: status={result.status_code}, request_id={result.request_id}")

    # 构造公网 URL: https://{bucket}.{endpoint_host}/{key}
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://{bucket}.{host}/{object_key}"

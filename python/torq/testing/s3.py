try:
    import boto3
    import botocore
except ImportError:
    boto3 = None


import os

from filelock import FileLock


S3_HF_CACHE_BUCKET_NAME = os.environ.get("S3_HF_CACHE_BUCKET_NAME", "")
S3_HF_CACHE_DIR = os.environ.get("S3_HF_CACHE_DIR", "")


_cache_upload_disabled = False


def get_file(cache, category, etag):
    """
    Downloads a file from the S3 bucket S3_HF_CACHE_BUCKET_NAME in directory S3_HF_CACHE_DIR to
    the local cache directory under cache_dir/s3_cache/category/etag.
    """

    cache_dir = cache.mkdir('s3_cache')

    local_dir = os.path.join(cache_dir, category)
    os.makedirs(local_dir, exist_ok=True)

    with FileLock(os.path.join(local_dir, f"{etag}.lock")):

        local_file_path = os.path.join(local_dir, etag)
        s3_key = os.path.join(S3_HF_CACHE_DIR, category, etag)

        if os.path.exists(local_file_path):
            return local_file_path
                
        if boto3 is None:
            return None

        if S3_HF_CACHE_BUCKET_NAME == "" or S3_HF_CACHE_DIR == "":
            return None

        s3 = boto3.client('s3')

        partial_file = local_file_path + ".partial"

        print(f"Downloading {s3_key} from S3 to {local_file_path}...")

        try:
            s3.download_file(S3_HF_CACHE_BUCKET_NAME, s3_key, partial_file)

            os.rename(partial_file, local_file_path)

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"File {s3_key} does not exist in S3.")
                return None
            else:
                raise
        except Exception as e:            
            raise

    return local_file_path


def put_file(cache, category, etag, local_file_path):
    """
    Uploads a file from the local cache directory under cache_dir/s3_cache/category/etag to
    the S3 bucket S3_HF_CACHE_BUCKET_NAME in directory S3_HF_CACHE_DIR if it is not already present.
    """

    global _cache_upload_disabled

    if _cache_upload_disabled:
        return local_file_path

    if boto3 is None:
        _cache_upload_disabled = True
        return local_file_path
    
    elif S3_HF_CACHE_BUCKET_NAME == "" or S3_HF_CACHE_DIR == "":
        _cache_upload_disabled = True
        return local_file_path    

    cache_dir = cache.mkdir('s3_cache')
        
    local_dir = os.path.join(cache_dir, category)

    os.makedirs(local_dir, exist_ok=True)

    # lock to prevent multiple uploads of the same file in parallel
    with FileLock(os.path.join(local_dir, f"{etag}.lock")):

        s3_key = os.path.join(S3_HF_CACHE_DIR, category, etag)

        # Check if file already exists in S3 before uploading
        should_upload = False    

        s3 = boto3.client('s3')    

        try:
            s3.head_object(Bucket=S3_HF_CACHE_BUCKET_NAME, Key=s3_key)
            print("File already exists in S3, skipping upload.")    

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':                
                should_upload = True
            else:
                raise

        if should_upload:    
            print(f"Uploading {local_file_path} to S3...")

            # multiple clients may race to upload the same file, but we can afford this for the moment
            try:
                s3.upload_file(local_file_path, S3_HF_CACHE_BUCKET_NAME, s3_key)
            except boto3.exceptions.S3UploadFailedError as e:
                _cache_upload_disabled = True

                print("Upload failed:", e)
                return local_file_path
        
        # create a local symlink in the s3 cache as well so that get_file can find it next time
        cached_file_path = os.path.join(local_dir, etag)

        if not os.path.exists(cached_file_path):
            print(f"Caching file locally at {cached_file_path}...")
            os.symlink(local_file_path, cached_file_path)

        return cached_file_path


            

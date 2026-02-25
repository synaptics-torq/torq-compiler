from django.contrib.auth.models import User
import os


def provision_admin_user():

    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        user = User.objects.create(username='admin', email='no-reply@synaptics.com', is_superuser=True, is_staff=True)        

    password = os.environ.get("ADMIN_PASSWORD", "password")
    user.set_password(password)
    user.save()

    print("Admin user 'admin' provisioned")


def provision_s3_bucket():
    # this is used only for local development with SeaweedFS, in production the bucket should be pre-provisioned
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        region_name=os.environ.get('AWS_REGION')
    )

    bucket_name = os.environ.get("S3_DATA_BUCKET")

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists")
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created")
        else:
            print(f"Error checking/creating bucket: {e}")
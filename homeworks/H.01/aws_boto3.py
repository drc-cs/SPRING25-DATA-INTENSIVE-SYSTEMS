"""
Module to interact with AWS S3 using boto3.
"""

import boto3

# Create a session using the default profile
session = boto3.Session(profile_name='default')

# Create a client
client = session.client('s3')

def get_all_buckets():
    """
    Get all buckets.

    Returns:
        dict: A dictionary containing the list of all buckets.
    """
    response = client.list_buckets()
    return response

def download_file(bucket, key, filename):
    """
    Download a file from S3.

    Args:
        bucket (str): The name of the bucket.
        key (str): The key of the file to download.
        filename (str): The local filename to save the downloaded file.
    """
    client.download_file(bucket, key, filename)

def upload_file(bucket, key, filename):
    """
    Upload a file to S3.

    Args:
        bucket (str): The name of the bucket.
        key (str): The key of the file to upload.
        filename (str): The local filename of the file to upload.
    """
    client.upload_file(filename, bucket, key)

def delete_file(bucket, key):
    """
    Delete a file from S3.

    Args:
        bucket (str): The name of the bucket.
        key (str): The key of the file to delete.
    """
    client.delete_object(Bucket=bucket, Key=key)

def create_bucket(bucket):
    """
    Create a new bucket in S3.

    Args:
        bucket (str): The name of the bucket to create.
    """
    client.create_bucket(Bucket=bucket)

def delete_bucket(bucket):
    """
    Delete a bucket from S3.

    Args:
        bucket (str): The name of the bucket to delete.
    """
    client.delete_bucket(Bucket=bucket)
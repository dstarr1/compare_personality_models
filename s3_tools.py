'''
This contains methods that interact with S3.
'''
import boto3
import gzip
from io import BytesIO
import os
import shutil
from urllib.parse import urlparse


def list_s3_recursivly(client=None, s3_subpath=None, bucket='your_bucket'):
    '''This recursivly lists files under an S3 path, also use pagination
     to allow for > 1000 objects.
    '''
    if client is None:
        client = boto3.client('s3')

    paginator = client.get_paginator('list_objects')
    path_list = []
    for result in paginator.paginate(Bucket=bucket,
                                     Delimiter='/',
                                     Prefix=s3_subpath):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                paths = list_s3_recursivly(client=client,
                                           s3_subpath=subdir.get('Prefix'),
                                           bucket=bucket)
                path_list.extend(paths)
        for file in result.get('Contents', []):
            path_str = "%s/%s" % (bucket, file.get('Key'))
            path_list.append(path_str)
    return path_list


def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
    ''' This downloads a S3 path to a local directory.
    It copies recursivly under path and use pagination to allow for
    > 1000 objects.
    '''
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket,
                                     Delimiter='/',
                                     Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client,
                             resource,
                             subdir.get('Prefix'),
                             local,
                             bucket)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(
                local, file.get('Key').replace(dist, ''))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            resource.meta.client.download_file(
                bucket, file.get('Key'), dest_pathname)


def parse_s3_dir(s3_dir):
    '''
    '''
    if "s3://" not in s3_dir:
        s3_dir = "s3://" + s3_dir
    urlobj = urlparse(s3_dir, allow_fragments=False)
    s3_bucket = urlobj.netloc
    s3_subpath = urlobj.path.lstrip('/')
    return (s3_bucket, s3_subpath)


def copy_directory(model_s3_dir, build_dirpath):
    ''' Retrieve a S3 folder/directory into a local path.
    '''
    (s3_bucket, s3_subpath) = parse_s3_dir(model_s3_dir)
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    download_dir(client,
                 resource,
                 s3_subpath,
                 local=build_dirpath,
                 bucket=s3_bucket)


def upload_str_uncompressed(data_str, bucket, obj_key):
    ''' String data Upload for smaller data, not using multipart, not
    storing gzip compressed.
    '''
    client = boto3.client('s3')
    client.put_object(Body=data_str, Bucket=bucket, Key=obj_key)


def upload_str_gzipped(data_str, bucket, obj_key):
    ''' Upload string data compressing with gzip compression to save
    network, storage costs.

    NOTE gzip requires whole file to compress and thus file must
    be loaded into memory.

    TODO: For files > memory, break into multipart & gzip, using:
      https://stackoverflow.com/questions/15754610/how-to-gzip-while-uploading-into-s3-using-boto
    '''
    uncompressed_content_type = 'application/json'  # 'text/plain'
    json_string_encoding = 'utf-8'
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)

    if type(data_str) == type(b''):
        # byte, API request.data case
        fp = BytesIO(data_str)
    else:
        # string, json from file case
        fp = BytesIO(
            data_str.encode(json_string_encoding))  # encode() not streaming
    fp.seek(0)
    compressed_fp = BytesIO()
    with gzip.GzipFile(fileobj=compressed_fp, mode='wb') as gz:
        shutil.copyfileobj(fp, gz)
    compressed_fp.seek(0)
    bucket.upload_fileobj(
        compressed_fp,
        obj_key,
        {'ContentType': uncompressed_content_type,
         'ContentEncoding': 'gzip'})


def upload_json(json_str, s3_dir, a_time, use_gzip=True):
    ''' Upload json to S3 bucket.
    '''
    (s3_bucket, s3_subpath) = parse_s3_dir(s3_dir)
    if use_gzip:
        # Default case:
        s3_obj_key = f"{s3_subpath}/{a_time}.json.gz"
        upload_str_gzipped(json_str, s3_bucket, s3_obj_key)
    else:
        s3_obj_key = f"{s3_subpath}/{a_time}.json"
        upload_str_uncompressed(json_str, s3_bucket, s3_obj_key)

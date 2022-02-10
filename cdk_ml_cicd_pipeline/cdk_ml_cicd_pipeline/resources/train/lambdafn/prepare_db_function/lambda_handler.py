import json
import boto3
import os
import ast
import pymysql
import base64
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    
    try:
        # secret情報を取得
        secret = get_secret()
        # テーブルを作成
        body = create_table(secret)

    except Exception as e:
        import traceback
        body = traceback.format_exc()

    return {
        'body': json.dumps(body)
    }

def get_secret():
    '''Secrets Mangerに接続してSecret情報を取得する
    '''

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=os.environ['REGION_NAME']
    )
    try:
    
        get_secret_value_response = client.get_secret_value(
            SecretId=os.environ['SECRETS_NAME']
        )

    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        else :
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            secret = base64.b64decode(get_secret_value_response['SecretBinary'])
        # strをdictに変換して返却
 
        secret = ast.literal_eval(secret)
        
        
        return secret

def create_table(secret):
    '''RDSに接続してテーブルを作成する
    '''
    passwd = secret['password']
    username = secret['username']
    host = secret['host']
    db_name = secret['dbInstanceIdentifier']
    
    try:
        conn = pymysql.connect(host=host, user=username, passwd=passwd)
        cur = conn.cursor()
        query = get_query()
    except Exception as e:
        print('Databse connection failed due to {}'.format(e))
        raise e
        
    result = ''

    for sql in query:
        try:
            cur.execute(sql)
            result += 'Success' 
            result += '\n' 
        except Exception as e:
            if e.args[0] == 1007:
                # DBがすでに作成済みの場合
                result += str(e.args[1]) 
                result += '\n' 
                continue
            if e.args[0] == 1050:
                # テーブルがすでに作成済みの場合
                result += str(e.args[1]) 
                result += '\n' 
                continue
            else:
                raise e
    return result
        
        
def get_query():
    with open('/opt/sql/prepare_db.sql', 'r') as f:
        return f.read().splitlines()
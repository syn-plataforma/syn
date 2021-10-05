import os

from pymongo import MongoClient, errors

from app.config.config import DevelopmentConfig


def main():
    try:
        mongo_api_host = os.environ.get('MONGO_API_HOST', DevelopmentConfig.MONGO_API_HOST)
        if os.environ.get('ARCHITECTURE', DevelopmentConfig.ARCHITECTURE) == 'codebooks':
            mongo_api_host = os.environ.get('MONGO_CODEBOOKS_API_HOST', DevelopmentConfig.MONGO_API_HOST)

        client = MongoClient(
            host=mongo_api_host,
            port=int(os.environ.get('MONGO_API_PORT', DevelopmentConfig.MONGO_API_PORT)),
            username=os.environ.get('MONGO_API_USER', DevelopmentConfig.MONGO_API_USER),
            password=os.environ.get('MONGO_API_PASSWORD', DevelopmentConfig.MONGO_API_PASSWORD),
            authSource=os.environ.get('MONGO_API_AUTH_SOURCE', DevelopmentConfig.MONGO_API_AUTH_SOURCE),
            authMechanism=os.environ.get('MONGO_API_AUTH_MECHANISM', DevelopmentConfig.MONGO_API_AUTH_MECHANISM)
        )
        client.server_info()
    except errors.ServerSelectionTimeoutError as err:
        print(err)


if __name__ == '__main__':
    main()

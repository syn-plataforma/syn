import os

from app.utils.database.MongoCollection import MongoCollection
from app.config.config import DevelopmentConfig


class NetBeansClearMongoCollection(MongoCollection):
    def __init__(
            self,
            # uri = "mongodb://user:password@example.com:port/?authSource=the_database&authMechanism=SCRAM-SHA-256"
            uri=f"mongodb://"
                f"{os.environ.get('MONGO_DATA_USER', DevelopmentConfig.MONGO_DATA_USER)}"
                f":{os.environ.get('MONGO_DATA_PASSWORD', DevelopmentConfig.MONGO_DATA_PASSWORD)}"
                f"@{os.environ.get('MONGO_DATA_HOST', DevelopmentConfig.MONGO_DATA_HOST)}"
                f":{os.environ.get('MONGO_DATA_PORT', DevelopmentConfig.MONGO_DATA_PORT)}/"
                f"?authSource={os.environ.get('MONGO_DATA_AUTH_SOURCE', DevelopmentConfig.MONGO_DATA_AUTH_SOURCE)}"
                f"&authMechanism={os.environ.get('MONGO_DATA_AUTH_MECHANISM', DevelopmentConfig.MONGO_DATA_AUTH_MECHANISM)}",
            db_name=os.environ.get('MONGO_DATA_NETBEANS_DATABASE', DevelopmentConfig.MONGO_DATA_NETBEANS_DATABASE),
            coll_name=os.environ.get('MONGO_DATA_COLLECTION', DevelopmentConfig.MONGO_DATA_COLLECTION)
    ):
        super().__init__(uri, db_name, coll_name)

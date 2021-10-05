import os

from app.utils.database.MongoCollection import MongoCollection
from app.config.config import DevelopmentConfig


class TrainingParametersMongoCollection(MongoCollection):
    mongo_api_host = os.environ.get('MONGO_API_HOST', DevelopmentConfig.MONGO_API_HOST)
    if os.environ.get('ARCHITECTURE', DevelopmentConfig.ARCHITECTURE) == 'codebooks':
        mongo_api_host = os.environ.get('MONGO_CODEBOOKS_API_HOST', DevelopmentConfig.MONGO_API_HOST)

    def __init__(
            self,
            # uri = "mongodb://user:password@example.com:port/?authSource=the_database&authMechanism=SCRAM-SHA-256"
            uri=f"mongodb://"
                f"{os.environ.get('MONGO_API_USER', DevelopmentConfig.MONGO_API_USER)}"
                f":{os.environ.get('MONGO_API_PASSWORD', DevelopmentConfig.MONGO_API_PASSWORD)}"
                f"@{mongo_api_host}"
                f":{os.environ.get('MONGO_API_PORT', DevelopmentConfig.MONGO_API_PORT)}/"
                f"?authSource={os.environ.get('MONGO_API_AUTH_SOURCE', DevelopmentConfig.MONGO_API_AUTH_SOURCE)}"
                f"&authMechanism={os.environ.get('MONGO_API_AUTH_MECHANISM', DevelopmentConfig.MONGO_API_AUTH_MECHANISM)}",
            db_name=os.environ.get('MONGO_API_DATABASE', DevelopmentConfig.MONGO_API_DATABASE),
            coll_name=os.environ.get('MONGO_API_TRAINING_PARAMETERS_COLLECTION',
                                     DevelopmentConfig.MONGO_API_TRAINING_PARAMETERS_COLLECTION)
    ):
        super().__init__(uri, db_name, coll_name)

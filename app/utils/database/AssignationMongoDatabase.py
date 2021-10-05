import os

from app.utils.database.MongoDatabase import MongoDatabase
from app.config.config import DevelopmentConfig


class AssignationMongoDatabase(MongoDatabase):
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
            db_name=os.environ.get('MONGO_DATA_ASSIGNATION_DATABASE',
                                   DevelopmentConfig.MONGO_DATA_ASSIGNATION_DATABASE),
            occu_coll_name=os.environ.get('MONGO_DATA_OCCUPATION_COLLECTION', DevelopmentConfig.MONGO_DATA_AUTH_SOURCE),
            adeq_coll_name=os.environ.get('MONGO_DATA_ADEQUACY_COLLECTION', DevelopmentConfig.MONGO_DATA_AUTH_SOURCE)
    ):
        super().__init__(uri, db_name)
        self.occupation = self.db[occu_coll_name]
        self.adequacy = self.db[adeq_coll_name]

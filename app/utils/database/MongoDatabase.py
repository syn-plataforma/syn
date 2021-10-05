from pymongo import MongoClient


class MongoDatabase(object):
    def __init__(
            self,
            uri,
            db_name
    ):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    @staticmethod
    def build_uri(user, password, host, port, auth_source, auth_mechanism):
        # uri = "mongodb://user:password@example.com:port/?authSource=the_database&authMechanism=SCRAM-SHA-256"
        return f"mongodb://" \
               f"{user}:{password}" \
               f"@{host}/" \
               f"?authSource={auth_source}" \
               f"&authMechanism={auth_mechanism}"

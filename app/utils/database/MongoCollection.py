from app.utils.database.MongoDatabase import MongoDatabase


class MongoCollection(MongoDatabase):
    def __init__(
            self,
            uri,
            db_name,
            coll_name
    ):
        super().__init__(uri, db_name)
        self.collection = self.db[coll_name]

    def insert_one(self, document):
        return self.collection.insert_one(document)

    def insert_many(self, documents):
        return self.collection.insert_many(documents)

    @staticmethod
    def mongodb_result_to_dict(query_result):
        output = []

        for result in query_result:
            item = {}
            try:
                for k, v in result.items():
                    if k != "_id":
                        item[k] = v
                output.append(item)
            except AttributeError as ae:
                return query_result

        return output

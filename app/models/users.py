from passlib.hash import pbkdf2_sha256 as sha256

from app.utils.database.database import get_api_user_collection


class User(object):
    def __init__(self, username, password, is_verified, email):
        self.username = username
        self.password = password
        self.is_verified = is_verified
        self.email = email

    def create(self):
        result = get_api_user_collection().insert_one(self)
        return self

    def update_field(self, field_name, field_value):
        result = get_api_user_collection().update(
            {'email': self.email}, {'$set': {str(field_name): field_value}}
        )
        return self

    @classmethod
    def find_by_email(cls, email):
        return get_api_user_collection().find_one({"email": email}, {"_id": 0})

    @classmethod
    def find_by_username(cls, username):
        return get_api_user_collection().find_one({"username": username}, {"_id": 0})

    @staticmethod
    def generate_hash(password):
        return sha256.hash(password)

    @staticmethod
    def verify_hash(password, hash):
        return sha256.verify(password, hash)

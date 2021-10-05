import json


class BaseParameters:

    def to_json(self):
        return json.dumps(self, indent=4, default=lambda o: o.__dict__)

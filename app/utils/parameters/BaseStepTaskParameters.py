from app.utils.parameters.BaseParameters import BaseParameters
from app.utils.parameters.ModelDumpParameters import ModelDumpParameters


class BaseStepTaskParameters(BaseParameters):

    def __init__(
            self,
            name='nlp',
            model_dump=None,
            execution_time_in_minutes=0.0,
            endpoint='nlp/',
            query_string=''
    ):
        self.name = name
        self.model_dump = []
        if model_dump is None:
            model_dump = []
        for dump in model_dump:
            if isinstance(dump, ModelDumpParameters):
                self.model_dump.append(dump)
                continue
            if isinstance(dump, dict):
                self.model_dump.append(ModelDumpParameters(dump['_id'], dump['_type']))

        self.execution_time_in_minutes = execution_time_in_minutes
        self.endpoint = endpoint
        self.query_string = query_string

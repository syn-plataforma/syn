from app.utils.parameters.BaseStepTaskParameters import BaseStepTaskParameters
from app.utils.parameters.ModelDumpParameters import ModelDumpParameters
from app.utils.parameters.RetrievalStepQueryStringParameters import RetrievalStepQueryStringParameters


class RetrievalStepTaskParameters(BaseStepTaskParameters):

    def __init__(
            self,
            name='retrieval',
            model_dump=None,
            execution_time_in_minutes=0.0,
            endpoint='retrieval/',
            query_string='',
            query_string_parameters=RetrievalStepQueryStringParameters()
    ):
        if model_dump is None:
            model_dump = [ModelDumpParameters()]
        super().__init__(name, model_dump, execution_time_in_minutes, endpoint, query_string)
        self.query_string_parameters = query_string_parameters


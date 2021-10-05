from app.utils.parameters.BaseParameters import BaseParameters


class ModelDumpParameters(BaseParameters):

    def __init__(
            self,
            _id='',
            _type=''

    ):
        self._id = _id
        self._type = _type

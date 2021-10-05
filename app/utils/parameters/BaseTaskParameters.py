from app.utils.parameters.BaseParameters import BaseParameters


class BaseTaskParameters(BaseParameters):

    def __init__(
            self,
            task_type='',
            task_name='',
            task_id=0,
            task_description='',
            task_corpus='Bugzilla'
    ):
        self.task_type = task_type
        self.task_name = task_name
        self.task_id = task_id
        self.task_description = task_description
        self.task_corpus = task_corpus

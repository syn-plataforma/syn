from app.utils.parameters.BaseTaskParameters import BaseTaskParameters
from app.utils.parameters.NlpStepTaskParameters import NlpStepTaskParameters
from app.utils.parameters.EmbeddingsStepTaskParameters import EmbeddingsStepTaskParameters
from app.utils.parameters.CodebooksStepTaskParameters import CodebooksStepTaskParameters
from app.utils.parameters.VectorizerStepTaskParameters import VectorizerStepTaskParameters
from app.utils.parameters.DuplicatesStepTaskParameters import DuplicatesStepTaskParameters


class DuplicatesTaskParameters(BaseTaskParameters):

    def __init__(
            self,
            task_type='duplicates',
            task_name='Bugzilla-DecisionTreeClassifier',
            task_id=0,
            task_description='Detección de incidencias duplicadas',
            task_corpus='Bugzilla',
            steps=None
    ):
        if steps is None:
            steps = [
                NlpStepTaskParameters(),
                EmbeddingsStepTaskParameters(),
                CodebooksStepTaskParameters(),
                VectorizerStepTaskParameters(),
                DuplicatesStepTaskParameters(),
                DuplicatesStepTaskParameters(name='all_steps', endpoint='all_duplicates/')
            ]
        super().__init__(task_type, task_name, task_id, task_description, task_corpus)
        self.steps = steps

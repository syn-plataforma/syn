from app.utils.parameters.BaseTaskParameters import BaseTaskParameters
from app.utils.parameters.AssignationStepParameters import AssignationStepParameters


class AssignationTaskParameters(BaseTaskParameters):

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
                AssignationStepParameters()
            ]
        super().__init__(task_type, task_name, task_id, task_description, task_corpus)
        self.steps = steps

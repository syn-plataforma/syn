from app.utils.parameters.BaseParameters import BaseParameters


class AssignationStepQueryStringParameters(BaseParameters):

    def __init__(
            self,
            topk=5,
            topk_text='Número de desarrolladores propuestos para asignarles la incidencia',
            occup_wheight=0.1,
            occup_wheight_text='Factor por el que se multiplica la ocupación del desarrollador'
    ):
        self.topk = topk
        self.topk_text = topk_text
        self.occup_wheight = occup_wheight
        self.occup_wheight_text = occup_wheight_text

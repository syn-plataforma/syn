/* eslint-disable @typescript-eslint/naming-convention */

import { AssignationPredictResponse } from './assignationPredict';

/**
 * Modelo que se pasa entre componentes con el trataimento y el entrenamiento que se usa
 */
export class IndexToTourModel {
  constructor(
    public treatment: string,
    public idTask: string,
    public corpus?: string,
    public method?: string
  ) {}
}

export interface IndexToForm {
  task_id: string;
  task_type: string;
  corpus: string;
}
export interface InputToDisplay {
  result: AssignationPredictResponse;
  task_type: string;
  incidencia: any;
}

export interface DataToInput {
  data?: IndexToForm;
  formsFields?: any;
  component?: any;
  bug_severity?: any;
  priority?: any;
  product?: any;
  completo: string;
}

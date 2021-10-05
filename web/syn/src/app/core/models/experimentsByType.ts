/* eslint-disable @typescript-eslint/naming-convention */

export interface ExperimentByType {
  task_corpus: string;
  task_description: string;
  task_id: string;
}

export interface ExperimentsByTypeResponse {
  code: string;
  result: ExperimentByType[];
}

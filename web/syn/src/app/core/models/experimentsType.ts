/* eslint-disable @typescript-eslint/naming-convention */
export interface ExperimentsType {
  task: string;
  task_name: string;
}

export interface ExperimentsTypeResponse {
  code: string;
  result: ExperimentsType[];
}

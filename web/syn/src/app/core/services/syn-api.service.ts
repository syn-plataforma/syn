/* eslint-disable @typescript-eslint/naming-convention */
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';

import { ExperimentsTypeResponse } from '../models/experimentsType';
import { ExperimentsByTypeResponse } from '../models/experimentsByType';
import { CorpusByTask } from '../models/corpus';
import { FormsFieldByTask } from '../models/formsFields';
import { IncidenciaRegistrada } from '../models/incidenciaRegistrada';
import { MetricsByTask } from '../models/metricsByTask';

const STEPS = '/steps';

/**
 * Servicio que contiene las peticiones al api de SYN.
 */
@Injectable({
  providedIn: 'root',
})
export class SynApiService {
  constructor(private http: HttpClient) {}

  /**
   * Petición que trae los entrenamientos.
   *
   */
  getTypeOfExperiments(): Observable<ExperimentsTypeResponse> {
    return this.http.get<ExperimentsTypeResponse>(
      environment.apiUri + '/experiments/tasks/'
    );
  }
  /**
   * Petición que trae los entrenamientos.
   *
   */
  getExperimentsByType(type: string): Observable<ExperimentsByTypeResponse> {
    return this.http.get<ExperimentsByTypeResponse>(
      environment.apiUri + '/experiments/task/' + `${type}/`
    );
  }

  getCorpus(task: string): Observable<CorpusByTask> {
    return this.http.get<CorpusByTask>(
      environment.apiUri + '/experiments/task/' + `${task}/corpus/`
    );
  }
  getExperimentsByTypeAndCorpus(task: string, corpus: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri + '/experiments/task/' + `${task}/corpus/${corpus}/`
    );
  }

  getFormsFieldsByTask(task: string): Observable<FormsFieldByTask> {
    return this.http.get<FormsFieldByTask>(
      environment.apiUri + '/features/task/' + `${task}/`
    );
  }
  getDataforFormsFields(field: string, corpus: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri + `/features/${field}/corpus/${corpus}/`
    );
  }
  modelHyperParameters(task_id: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri + `/experiments/${task_id}/model/hyperparameters/`
    );
  }
  modelMetrics(task_id: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri + `/experiments/${task_id}/model/metrics/`
    );
  }
  modelPredict(
    task_id: string,
    incidenciaRegistrada: IncidenciaRegistrada
  ): Observable<any> {
    return this.http.post<any>(
      environment.apiUri + `/experiments/${task_id}/model/predict/`,
      incidenciaRegistrada.transform()
    );
  }
  metricsByTask(task_type: string): Observable<MetricsByTask> {
    return this.http.get<MetricsByTask>(
      environment.apiUri + `/metrics/task/${task_type}/`
    );
  }
  taskByCorpus(corpus: string): Observable<any> {
    return this.http.get<any>(environment.apiUri + `/dataset/${corpus}/tasks/`);
  }
  bestModelByTaskAndCorpus(task_type: string, corpus: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri +
        `/experiments/task/${task_type}/corpus/${corpus}/best-model/`
    );
  }

  getDatasetStatics(task_type: string, corpus: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri +
        `/dataset/statistics/task/${task_type}/corpus/${corpus}/`
    );
  }
  getMetricsNameByTask(task_type: string): Observable<any> {
    return this.http.get<any>(
      environment.apiUri + `/metrics/task/${task_type}/`
    );
  }

  getAggregatedMetricByTaskAndCorpus(
    task_type: string,
    corpus: string
  ): Observable<any> {
    return this.http.get<any>(
      environment.apiUri +
        `/experiments/task/${task_type}/corpus/${corpus}/aggregated-metrics/`
    );
  }
}

/* eslint-disable guard-for-in */
import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { Router } from '@angular/router';
import { TranslateService } from '@ngx-translate/core';
import { corpusData } from 'src/app/core/data/corpus.data';
import { SharingdataService } from 'src/app/core/services/sharingdata.service';
import { SynApiService } from 'src/app/core/services/syn-api.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent implements OnInit {
  corpusList = corpusData;
  corpus: string;
  taskType: string;
  bestModels = [];
  listTask = [];
  modelPrio: any;
  modelClas: any;
  modelDupli: any;
  modelSimil: any;
  modelAssig: any;
  modelCustomAssig: any;
  metricasBestModels: any;
  metricsDupAndSim = [];
  metricsPrioAndAssigAndClass = [];
  getDatos: string;

  constructor(
    private formBuilder: FormBuilder,
    private router: Router,
    private sharingService: SharingdataService,
    private synService: SynApiService,
    private changeDetectorRefs: ChangeDetectorRef,
    public translate: TranslateService
  ) {}

  ngOnInit(): void {
    this.getDatos = '';
    this.metricasBestModels = {};
  }

  async loadDashboard() {
    this.getDatos = '';
    this.bestModels = [];
    this.listTask = [];
    this.synService.taskByCorpus(this.corpus).subscribe(async (result) => {
      console.log(result);
      this.listTask = result.result;

      for (const task of this.listTask) {
        await this.synService
          .bestModelByTaskAndCorpus(task, this.corpus)
          .toPromise()
          .then((model) => {
            console.log(model.result[0].task_id);
            if (task === 'custom_assignation') {
              this.modelCustomAssig = model.result[0].task_id;
              this.bestModels.push(this.modelCustomAssig);
              this.metricasBestModels[task] = {};
              this.metricasBestModels[task].id = model.result[0].task_id;
            } else if (task === 'prioritization') {
              this.modelPrio = model.result[0].task_id;
              this.bestModels.push(this.modelPrio);
              this.metricasBestModels[task] = {};
              this.metricasBestModels[task].id = model.result[0].task_id;
            } else if (task === 'classification') {
              this.modelClas = model.result[0].task_id;
              this.bestModels.push(this.modelClas);
              this.metricasBestModels[task] = {};
              this.metricasBestModels[task].id = model.result[0].task_id;
            } else if (task === 'duplicity') {
              this.modelDupli = model.result[0].task_id;
              this.bestModels.push(this.modelDupli);
              this.metricasBestModels[task] = {};
              this.metricasBestModels[task].id = model.result[0].task_id;
            } else if (task === 'similarity') {
              this.modelSimil = model.result[0].task_id;
              this.bestModels.push(this.modelSimil);
              this.metricasBestModels[task] = {};
              this.metricasBestModels[task].id = model.result[0].task_id;
            } else if (task === 'assignation') {
              this.modelAssig = model.result[0].task_id;
              this.bestModels.push(this.modelAssig);
              this.metricasBestModels[task] = {};
              this.metricasBestModels[task].id = model.result[0].task_id;
            }
          });
      }
      console.log(this.bestModels);
      console.log(this.metricasBestModels);

      console.log(this.metricsDupAndSim);
      console.log(this.metricsPrioAndAssigAndClass);

      for (const task of this.listTask) {
        if (task !== undefined) {
          await this.synService
            .getMetricsNameByTask(task)
            .toPromise()
            .then((metrics) => {
              if (
                task === 'custom_assignation' &&
                this.metricsPrioAndAssigAndClass.length === 0
              ) {
                this.metricsPrioAndAssigAndClass.push(metrics.result);
              } else if (
                task === 'prioritization' &&
                this.metricsPrioAndAssigAndClass.length === 0
              ) {
                this.metricsPrioAndAssigAndClass.push(metrics.result);
              } else if (
                task === 'classification' &&
                this.metricsPrioAndAssigAndClass.length === 0
              ) {
                this.metricsPrioAndAssigAndClass.push(metrics.result);
              } else if (
                task === 'duplicity' &&
                this.metricsDupAndSim.length === 0
              ) {
                this.metricsDupAndSim.push(metrics.result);
              } else if (
                task === 'similarity' &&
                this.metricsDupAndSim.length === 0
              ) {
                this.metricsDupAndSim.push(metrics.result);
              } else if (
                task === 'assignation' &&
                this.metricsPrioAndAssigAndClass.length === 0
              ) {
                this.metricsPrioAndAssigAndClass.push(metrics.result);
              }
            });
        }
      }
      for (const taskModelInfo in this.metricasBestModels) {
        if (this.metricasBestModels[taskModelInfo].id !== undefined) {
          await this.synService
            .modelMetrics(this.metricasBestModels[taskModelInfo].id)
            .toPromise()
            .then((bestModelMetrics) => {
              this.metricasBestModels[taskModelInfo].metricas =
                bestModelMetrics.result[0].metrics;
            });
        }
      }

      for (const task of this.listTask) {
        if (task !== undefined) {
          await this.synService
            .getDatasetStatics(task, this.corpus)
            .toPromise()
            .then((statics) => {
              console.log(this.metricasBestModels);
              console.log(statics);
              this.metricasBestModels[task] = {
                ...this.metricasBestModels[task],
                statics: statics.result,
              };
            });
        }
      }

      console.log(this.metricasBestModels);
      this.getDatos = 'COMPLETE';
    });
  }
}

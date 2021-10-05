/* eslint-disable @typescript-eslint/dot-notation */
/* eslint-disable @typescript-eslint/quotes */
/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable no-shadow */
/* eslint-disable no-trailing-spaces */
/* eslint-disable @typescript-eslint/semi */
import { DOCUMENT } from '@angular/common';
import { ChangeDetectorRef, Component, Inject, OnInit } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { TranslateService } from '@ngx-translate/core';
import { ExperimentByType } from 'src/app/core/models/experimentsByType';
import { ExperimentsType } from 'src/app/core/models/experimentsType';
import { SharingdataService } from 'src/app/core/services/sharingdata.service';
import { SynApiService } from 'src/app/core/services/syn-api.service';

@Component({
  selector: 'app-metrics',
  templateUrl: './metrics.component.html',
  styleUrls: ['./metrics.component.scss'],
})
export class MetricsComponent implements OnInit {
  idTask: string;
  corpus: string;
  taskType: string;
  listCorpusByTask: string[] = [];
  listExperiments: ExperimentsType[] = [];
  listExperimentsByType: ExperimentByType[] = [];
  isFilledTest = true;
  completeQuest = '';
  hyperParameters: any;
  metrics: any;
  constructor(
    @Inject(DOCUMENT) private document: HTMLDocument,
    private formBuilder: FormBuilder,
    private router: Router,
    private sharingService: SharingdataService,
    private synService: SynApiService,
    private changeDetectorRefs: ChangeDetectorRef,
    public translate: TranslateService,
    public dialog: MatDialog
  ) {}

  ngOnInit(): void {
    this.selectExperiment();
  }
  selectExperiment() {
    this.synService.getTypeOfExperiments().subscribe(
      (result) => {
        this.listExperiments = result.result;
      },
      (error) => {}
    );

    this.isFilledTest = false;
  }
  getCorpusByType(type: string) {
    this.synService.getCorpus(type).subscribe(
      (result) => {
        this.listCorpusByTask = result.result;
      },
      (error) => {}
    );
  }
  loadExperimentsByTypeAndCorpus(corpus: string) {
    this.synService
      .getExperimentsByTypeAndCorpus(this.taskType, corpus)
      .subscribe(
        (result) => {
          this.listExperimentsByType = result.result;
        },
        (error) => {}
      );
  }
  activateButton() {
    console.log('activar boton ');
    this.isFilledTest = true;
    console.log(this.isFilledTest);
    console.log(this.idTask);
  }
  getMetricsAndHyperparameters() {
    this.synService.modelHyperParameters(this.idTask).subscribe((result) => {
      this.hyperParameters = result.result[0].hyperparameters;
      this.synService.modelMetrics(this.idTask).subscribe((result) => {
        const aux = result.result[0].metrics;
        console.log(aux);
        delete aux['confusion_matrix'];
        this.metrics = aux;
      });
    });

    this.completeQuest = 'DONE';
  }
}

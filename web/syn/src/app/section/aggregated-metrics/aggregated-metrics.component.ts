/* eslint-disable @typescript-eslint/dot-notation */
import { Component, OnInit } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';
import { TableVirtualScrollDataSource } from 'ng-table-virtual-scroll';
import { corpusData } from 'src/app/core/data/corpus.data';
import { SynApiService } from 'src/app/core/services/syn-api.service';

@Component({
  selector: 'app-aggregated-metrics',
  templateUrl: './aggregated-metrics.component.html',
  styleUrls: ['./aggregated-metrics.component.scss'],
})
export class AggregatedMetricsComponent implements OnInit {
  task: string;
  corpus: string;
  corpusList = corpusData;
  taskList = [];
  filas = [];
  displayedColumns = [];
  done = '';
  constructor(
    private synService: SynApiService,
    public translate: TranslateService
  ) {}

  ngOnInit(): void {}
  loadTask() {
    this.synService.taskByCorpus(this.corpus).subscribe(
      (result) => {
        this.taskList = result.result;
      },
      (error) => {}
    );
  }

  async loadAggregatedMetrics() {
    this.filas = [];
    this.displayedColumns = [];
    this.done = '';
    this.synService
      .getAggregatedMetricByTaskAndCorpus(this.task, this.corpus)
      .subscribe((result) => {
        console.log(result);
        const headers = result.result.headers as Array<string>;
        const table = result.result.table as Array<any>;
        console.log(headers.includes('confusion_matrix'));
        if (headers.includes('confusion_matrix')) {
          const index = headers.indexOf('confusion_matrix');

          table.forEach((element) => {
            if (element.includes(NaN)) {
              element.forEach((list) => {
                if (isNaN(list)) {
                  element.splice(element.indexOf(list), 1);
                  element.add(element.indexOf(list), 'Nan');

                  console.log(element.splice(element.indexOf(list), 1));
                }
              });
            }
            element.splice(index, 1);
            element.splice(0, 1);
            console.log(element.splice(index, 1));
          });
          headers.splice(index, 1);
        }
        this.displayedColumns = headers;
        this.filas = table;
      });
    this.done = 'DONE';
  }
}

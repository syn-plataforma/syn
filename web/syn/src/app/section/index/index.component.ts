/* eslint-disable @typescript-eslint/ban-types */
/* eslint-disable @typescript-eslint/naming-convention */
import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { MatSelectChange } from '@angular/material/select';
import { TooltipPosition } from '@angular/material/tooltip';
import { Router } from '@angular/router';
import { TranslateService } from '@ngx-translate/core';
import { treatments } from 'src/app/core/data/treatments.data';
import { ExperimentByType } from 'src/app/core/models/experimentsByType';
import { ExperimentsType } from 'src/app/core/models/experimentsType';
import { IndexToTourModel } from 'src/app/core/models/indexTo.model';
import { SharingdataService } from 'src/app/core/services/sharingdata.service';
import { SynApiService } from 'src/app/core/services/syn-api.service';

/**
 * Componente "índice", es el que se muestra al cargar el app y muestra una breve info de la misma.
 *
 * Se encuentra contenido dentro de {@Link AppComponent}
 */
@Component({
  selector: 'app-index',
  templateUrl: './index.component.html',
  styleUrls: ['./index.component.scss'],
})
export class IndexComponent implements OnInit {
  idTask: string;
  corpus: string;
  taskType: string;
  listCorpusByTask: string[] = [];
  listExperiments: ExperimentsType[] = [];
  listExperimentsByType: ExperimentByType[] = [];

  displayedColumns: string[] = ['text', 'value'];
  isFilledTest = true;
  form: FormGroup;
  constructor(
    private formBuilder: FormBuilder,
    private router: Router,
    private sharingService: SharingdataService,
    private synService: SynApiService,
    private changeDetectorRefs: ChangeDetectorRef,
    public translate: TranslateService,
    public dialog: MatDialog
  ) {
    this.form = this.formBuilder.group({
      experiment: '',
    });
  }

  ngOnInit(): void {
    this.selectExperiment();
  }

  /**
   * Devuelve todas las tareas para el tratamiento seleccionado
   *
   * @param treatment Identificador del tratamiento seleccionado
   */
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

  /**
   * Método que redirecciona a la ruta seleccionada
   *
   * @param url Ruta seleccionada
   */
  changeComponent(url: string) {
    this.sharingService.setData({
      task_id: this.idTask,
      task_type: this.taskType,
      corpus: this.corpus,
    });

    this.router.navigate([url]); // redirects url to new component
  }
}

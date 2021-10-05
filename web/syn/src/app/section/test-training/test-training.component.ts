/* eslint-disable eqeqeq */
/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable no-underscore-dangle */
import { OnDestroy } from '@angular/core';
import { Component, OnInit, ViewChild } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Router } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { IncidenciaRegistrada } from 'src/app/core/models/incidenciaRegistrada';
import {
  DataToInput,
  IndexToForm,
  IndexToTourModel,
  InputToDisplay,
} from 'src/app/core/models/indexTo.model';
import { SharingdataService } from 'src/app/core/services/sharingdata.service';
import { SynApiService } from 'src/app/core/services/syn-api.service';
import { DisplayComponent } from './presenters/display/display.component';
import { InputTestComponent } from './presenters/input-test/input-test.component';
/**
 * Componente que se encargar de mostrar las vistas del "modo directo".
 *
 * Contenido por {@Link AppComponent}
 *
 */
@Component({
  selector: 'app-test-training',
  templateUrl: './test-training.component.html',
  styleUrls: ['./test-training.component.scss'],
})
export class TestTrainingComponent implements OnInit, OnDestroy {
  @ViewChild('stepOne') stepOneComponent: InputTestComponent;
  @ViewChild('stepTwo') stepTwoComponent: DisplayComponent;
  isEditable = false;
  listFormsFields: string[] = [];
  public showOverlay = true;
  inputToDisplay: InputToDisplay;
  dataFromIndex: IndexToForm;
  dataToInputPresenter: DataToInput;

  dataProduct: any;
  dataBugSeverity: any;
  dataComponent: any;
  dataPriority: any;
  dataMock: any;

  public tituloTratamiento: string;
  $isLoading: BehaviorSubject<boolean> = new BehaviorSubject(false);
  listaAux: string[];

  constructor(
    private router: Router,
    private synApiService: SynApiService,
    private sharingService: SharingdataService,
    private _snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.dataToInputPresenter = {
      product: [],
      priority: [],
      bug_severity: [],
      component: [],
      completo: '',
    };
    this.inputToDisplay = {
      result: { code: '', result: [] },
      task_type: '',
      incidencia: {},
    };
    this.dataToInputPresenter.formsFields = [];
    if (this.sharingService.getData() === '') {
      this.router.navigate(['']);
      this.openSnackBar();
    } else {
      this.dataFromIndex = this.sharingService.getData();
      console.log(this.dataFromIndex);
    }
    this.tituloTratamiento = this.dataFromIndex.task_type;
    this.loadFormsFields(this.dataFromIndex.task_type);
  }

  /**
   * Al cerrar el componente limpia los datos compartidos
   */
  ngOnDestroy() {
    this.sharingService.setData('');
    this.sharingService.setTable(new Array());
  }

  get frmStepOne() {
    return this.stepOneComponent ? this.stepOneComponent.frmStepOne$ : null;
  }
  get frmStepTwo() {
    return this.stepTwoComponent ? this.stepTwoComponent.frmStepTwo$ : null;
  }

  openSnackBar() {
    this._snackBar.open(
      'Debe de elegir un tratamiento antes de continuar',
      'Cerrar',
      {
        duration: 2000,
      }
    );
  }

  async loadFormsFields(task: string) {
    this.$isLoading.next(true);
    this.synApiService.getFormsFieldsByTask(task).subscribe(
      async (result) => {
        console.log(result);
        this.listFormsFields = result.result;
        this.listaAux = this.listFormsFields;
        this.removeItemFromList(this.listaAux, 'bug_id');
        this.removeItemFromList(this.listaAux, 'description');
        for (const element of this.listaAux) {
          await this.synApiService
            .getDataforFormsFields(element, this.dataFromIndex.corpus)
            .toPromise()
            .then(
              (data) => {
                this.dataMock = data.result;
              },
              (error) => {}
            );

          if (this.dataMock !== undefined) {
            if (element === 'component') {
              this.dataComponent = this.dataMock;
              this.dataToInputPresenter.component = this.dataComponent;
            } else if (element === 'product') {
              this.dataProduct = this.dataMock;
              this.dataToInputPresenter.product = this.dataProduct;
            } else if (element === 'priority') {
              this.dataPriority = this.dataMock;
              this.dataToInputPresenter.priority = this.dataPriority;
            } else {
              this.dataBugSeverity = this.dataMock;
              this.dataToInputPresenter.bug_severity = this.dataBugSeverity;
            }
            this.dataToInputPresenter.data = this.dataFromIndex;
            this.dataToInputPresenter.formsFields = this.listFormsFields;
          }
        }
        this.dataToInputPresenter.completo = 'DONE';
        this.$isLoading.next(false);
      },
      (error) => {}
    );
  }

  registrarIncidencia(incidenciaRegistrada: IncidenciaRegistrada) {
    // TODO IMPLEMENTAR LOS CASE PARA LOS DISTINTOS TRATAMIENTOS
    this.$isLoading.next(true);
    this.synApiService
      .modelPredict(this.dataFromIndex.task_id, incidenciaRegistrada)
      .subscribe((res) => {
        this.inputToDisplay = {
          result: res,
          task_type: this.dataFromIndex.task_type,
          incidencia: incidenciaRegistrada,
        };

        this.$isLoading.next(false);
      });
    console.log(incidenciaRegistrada);
  }
  removeItemFromList(features: any, item: string) {
    const i = features.indexOf(item);
    console.log(i);
    if (i != -1) {
      features.splice(i, 1);
    }
  }
}

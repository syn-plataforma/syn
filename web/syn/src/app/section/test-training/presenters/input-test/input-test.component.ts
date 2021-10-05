/* eslint-disable @typescript-eslint/semi */
/* eslint-disable @typescript-eslint/no-unused-expressions */
/* eslint-disable @angular-eslint/use-lifecycle-interface */
/* eslint-disable eqeqeq */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable @typescript-eslint/member-ordering */
import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnInit,
  Output,
  ChangeDetectionStrategy,
  AfterViewInit,
  AfterContentInit,
} from '@angular/core';
import {
  FormBuilder,
  FormGroup,
  ValidatorFn,
  Validators,
} from '@angular/forms';

import { BehaviorSubject, Observable } from 'rxjs';
import { delay, map, startWith, window } from 'rxjs/operators';
import {
  componentsDataEclipse,
  componentsDataNetbeans,
  componentsDataOpenOffice,
} from 'src/app/core/data/component.data';
import { corpusData } from 'src/app/core/data/corpus.data';
import { prioritiesData } from 'src/app/core/data/priority.data';
import {
  productsDataEclipse,
  productsDataNetbeans,
  productsDataOpenoffice,
} from 'src/app/core/data/product.data';
import { severities } from 'src/app/core/data/severity.data';
import { MyErrorTypes } from 'src/app/core/error/error.enum';
import { MyError } from 'src/app/core/error/error.model';
import { IncidenciaRegistrada } from 'src/app/core/models/incidenciaRegistrada';
import {
  DataToInput,
  IndexToForm,
  IndexToTourModel,
} from 'src/app/core/models/indexTo.model';

import { itemIncludedOn } from 'src/app/shared/shared.module';

@Component({
  selector: 'app-input-test',
  templateUrl: './input-test.component.html',
  styleUrls: ['./input-test.component.scss'],
})
export class InputTestComponent implements OnInit {
  @Output() public incidenciaEmitter = new EventEmitter<IncidenciaRegistrada>();
  @Input() data: DataToInput;
  frmStepOne: FormGroup;
  frmStepOne$: Observable<FormGroup>;
  bugPriorities: any = [];
  bugSeverities: any = [];
  components: any = [];
  products: any = [];
  submitted = true;
  corpus = corpusData;

  filteredOptionsComponent: Observable<string[]>;
  filteredOptionsProduct: Observable<string[]>;

  filteredOptions: Observable<string[]>;

  private myFrmStepOne$ = new BehaviorSubject<FormGroup>(null);
  myFrmStepOneListener$: Observable<FormGroup> =
    this.myFrmStepOne$.asObservable();

  myFrmStepOne(form: FormGroup) {
    this.myFrmStepOne$.next(form);
  }

  constructor(private formBuilder: FormBuilder) {}
  /**
   * Carga el formulario y los desplegables del componente
   */

  ngOnInit(): void {
    this.loadForm();
  }

  loadForm() {
    this.frmStepOne$ = this.myFrmStepOneListener$.pipe(delay(5000));
    console.log(this.data.data.task_type);
    console.log(this.data);
    try {
      switch (this.data.data.task_type) {
        case 'custom_assignation':
          this.components = this.data.component;
          this.products = this.data.product;
          this.bugPriorities = this.data.priority;
          this.bugSeverities = this.data.bug_severity;
          break;
        case 'assignation':
          this.components = this.data.component;
          this.products = this.data.product;
          this.bugPriorities = this.data.priority;
          this.bugSeverities = this.data.bug_severity;

          break;

        case 'classification':
          this.components = this.data.component;
          this.products = this.data.product;
          this.bugPriorities = this.data.priority;
          this.bugSeverities = this.data.bug_severity;

          break;

        case 'duplicity':
          this.components = this.data.component;
          this.products = this.data.product;
          this.bugPriorities = this.data.priority;
          this.bugSeverities = this.data.bug_severity;

          break;
        case 'prioritization':
          this.components = this.data.component;
          this.products = this.data.product;
          this.bugPriorities = this.data.priority;
          this.bugSeverities = this.data.bug_severity;

          break;
        case 'similarity':
          this.components = this.data.component;
          this.products = this.data.product;
          this.bugPriorities = this.data.priority;
          this.bugSeverities = this.data.bug_severity;

          break;
      }
      this.frmStepOne = this.formBuilder.group({
        product: ['', [itemIncludedOn(this.products)]],
        description: ['', []],
        bug_severity: ['', []],
        priority: ['', []],
        component: ['', [itemIncludedOn(this.components)]],
        bug_id: ['', []],
      });
      this.myFrmStepOne(this.frmStepOne);
      this.filteredOptionsComponent =
        this.frmStepOne.controls.component.valueChanges.pipe(
          startWith(''),
          map((value) => (typeof value === 'string' ? value : value.name)),
          map((name) =>
            name
              ? this._filterComponent(name, this.components)
              : this.components.slice()
          )
        );
      this.filteredOptionsProduct =
        this.frmStepOne.controls.product.valueChanges.pipe(
          startWith(''),
          map((value) => (typeof value === 'string' ? value : value.name)),
          map((name) =>
            name
              ? this._filterProduct(name, this.products)
              : this.products.slice()
          )
        );
    } catch (e) {
      new MyError(
        'Training no encontrado',
        `Para llegar a esta ruta debe de elegir un tratamiento y un entrenamiento desde el index`,
        MyErrorTypes.WARN
      ).show();
    }
  }
  buildRequest() {
    return new IncidenciaRegistrada(
      this.frmStepOne.controls.description.value,
      this.frmStepOne.controls.bug_id.value,
      this.frmStepOne.controls.product.value,
      this.frmStepOne.controls.bug_severity.value,
      this.frmStepOne.controls.priority.value,
      this.frmStepOne.controls.component.value
    );
  }

  displayFn(item: string): string {
    return item ? item : '';
  }
  /**
   * Método que filtra el producto introducido en la lista de productos
   *
   * @param name Nombre del producto
   * @param array Listado de productos
   */
  private _filterProduct(name: string, array: any[]): string[] {
    const filterValue = name.toLowerCase();

    return array.filter(
      (optionProduct) => optionProduct.toLowerCase().indexOf(filterValue) === 0
    );
  }
  /**
   * Método que filtra el componente introducido en la lista de componentes
   *
   * @param name Nombre del componente
   * @param array Lisado de componentes
   */
  private _filterComponent(name: string, array: any[]): string[] {
    const filterValue = name.toLowerCase();

    return array.filter(
      (optionComponent) =>
        optionComponent.toLowerCase().indexOf(filterValue) === 0
    );
  }
  get f() {
    return this.frmStepOne.controls;
  }
  /**
   * Método que valída y envía el formulario
   */
  emitForm(nuevaIncidencia: IncidenciaRegistrada) {
    this.incidenciaEmitter.emit(nuevaIncidencia);
  }
}

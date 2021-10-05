/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable @typescript-eslint/member-ordering */
import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { BehaviorSubject, Observable } from 'rxjs';
import { delay } from 'rxjs/operators';
import { IncidenciaRegistrada } from 'src/app/core/models/incidenciaRegistrada';
import { InputToDisplay } from 'src/app/core/models/indexTo.model';
import { SharingdataService } from 'src/app/core/services/sharingdata.service';

@Component({
  selector: 'app-display',
  templateUrl: './display.component.html',
  styleUrls: ['./display.component.scss'],
})
export class DisplayComponent implements OnInit {
  @Input() data: InputToDisplay;
  displayedColumns: string[] = ['texto', 'asignado', 'predicho'];
  aux: any;
  step = 'all_steps';
  frmStepTwo: FormGroup;
  frmStepTwo$: Observable<FormGroup>;
  tituloTratamiento: string;
  listResult = [];
  private myFrmStepTwo$ = new BehaviorSubject<FormGroup>(null);
  myFrmStepTwoListener$: Observable<FormGroup> = this.myFrmStepTwo$.asObservable();

  constructor(
    private formBuilder: FormBuilder,
    private sharingService: SharingdataService
  ) {
    this.frmStepTwo = this.formBuilder.group({
      address: [''],
    });
  }

  ngOnInit(): void {
    console.log(this.data);
    this.listResult = this.data.result.result;
    this.myFrmStepTwo(this.frmStepTwo);

    this.frmStepTwo$ = this.myFrmStepTwoListener$.pipe(delay(0));
  }
  /**
   * Cambia de componente
   *
   * @param form Formulario
   */
  myFrmStepTwo(form: FormGroup) {
    this.myFrmStepTwo$.next(form);
  }
}

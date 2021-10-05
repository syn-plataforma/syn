import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { SharedRoutingModule } from './shared-routing.module';
import { NavbarComponent } from './navbar/navbar.component';
import { FooterComponent } from './footer/footer.component';
import { NotFoundComponent } from './not-found/not-found.component';

import { MatButtonModule } from '@angular/material/button';
import { MatDividerModule } from '@angular/material/divider';
import { MatMenuModule } from '@angular/material/menu';
import { MatStepperModule } from '@angular/material/stepper';
import { MatToolbarModule } from '@angular/material/toolbar';
import { NgScrollbarModule } from 'ngx-scrollbar';
import { TranslateModule } from '@ngx-translate/core';
import { MatTableModule } from '@angular/material/table';
import { AbstractControl, ValidatorFn } from '@angular/forms';
import {
  MatProgressSpinnerModule,
  MatSpinner,
} from '@angular/material/progress-spinner';
import { MatDialogModule } from '@angular/material/dialog';

@NgModule({
  declarations: [NavbarComponent, FooterComponent, NotFoundComponent],
  imports: [
    CommonModule,
    SharedRoutingModule,
    MatTableModule,
    MatToolbarModule,
    MatButtonModule,
    MatMenuModule,
    NgScrollbarModule,
    MatDividerModule,
    MatStepperModule,
    MatProgressSpinnerModule,
    TranslateModule,
    MatDialogModule,
  ],
  exports: [
    TranslateModule,
    NavbarComponent,
    FooterComponent,
    MatProgressSpinnerModule,
    MatDialogModule,
  ],
})
export class SharedModule {}

export const itemIncludedOn = (array: Array<any>) => (
  control: AbstractControl
): { [key: string]: boolean } | null => {
  if (!array.includes(control.value)) {
    return { isItemNotIncluded: true };
  }
  return null;
};

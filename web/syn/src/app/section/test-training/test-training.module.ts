import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { TestTrainingRoutingModule } from './test-training-routing.module';
import { TestTrainingComponent } from './test-training.component';
import { InputTestComponent } from './presenters/input-test/input-test.component';
import { DisplayComponent } from './presenters/display/display.component';
import { MatStepperModule } from '@angular/material/stepper';
import { MatFormFieldModule } from '@angular/material/form-field';
import { ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatCardModule } from '@angular/material/card';
import { FlexLayoutModule } from '@angular/flex-layout';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { SharedModule } from 'src/app/shared/shared.module';
import {
  MatProgressSpinner,
  MatProgressSpinnerModule,
  MatSpinner,
} from '@angular/material/progress-spinner';

@NgModule({
  declarations: [TestTrainingComponent, InputTestComponent, DisplayComponent],
  imports: [
    CommonModule,
    SharedModule,
    TestTrainingRoutingModule,
    MatTableModule,
    MatStepperModule,
    MatFormFieldModule,
    ReactiveFormsModule,
    MatInputModule,
    MatSelectModule,
    MatExpansionModule,
    MatAutocompleteModule,
    MatButtonModule,
    MatCardModule,
    MatProgressSpinnerModule,
    FlexLayoutModule,
  ],
})
export class TestTrainingModule {}

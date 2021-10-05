import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { MetricsRoutingModule } from './metrics-routing.module';
import { FlexLayoutModule } from '@angular/flex-layout';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatTableModule } from '@angular/material/table';
import { MatTooltipModule } from '@angular/material/tooltip';
import { SharedModule } from 'src/app/shared/shared.module';
import { MetricsComponent } from './metrics.component';
import { TranslateModule } from '@ngx-translate/core';

@NgModule({
  declarations: [MetricsComponent],
  imports: [
    CommonModule,
    MetricsRoutingModule,
    SharedModule,
    MatButtonModule,
    MatCardModule,
    MatDividerModule,
    MatTableModule,
    MatSelectModule,
    FlexLayoutModule,
    FormsModule,
    TranslateModule,
    MatTooltipModule,
    MatIconModule,
  ],
})
export class MetricsModule {}

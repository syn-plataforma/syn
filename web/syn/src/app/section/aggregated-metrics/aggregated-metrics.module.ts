import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { AggregatedMetricsRoutingModule } from './aggregated-metrics-routing.module';
import { AggregatedMetricsComponent } from './aggregated-metrics.component';
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
import { TableVirtualScrollModule } from 'ng-table-virtual-scroll';
import { TranslateModule } from '@ngx-translate/core';
import { MatSidenavModule } from '@angular/material/sidenav';

@NgModule({
  declarations: [AggregatedMetricsComponent],
  imports: [
    CommonModule,
    SharedModule,
    AggregatedMetricsRoutingModule,
    MatButtonModule,
    MatCardModule,
    MatDividerModule,
    MatTableModule,
    MatSelectModule,
    TranslateModule,
    TableVirtualScrollModule,
    FlexLayoutModule,
    MatSidenavModule,
    FormsModule,
    MatTooltipModule,
    MatIconModule,
  ],
})
export class AggregatedMetricsModule {}

import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { IndexRoutingModule } from './index-routing.module';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';
import { IndexComponent } from './index.component';
import { MatSelectModule } from '@angular/material/select';
import { SharingdataService } from 'src/app/core/services/sharingdata.service';
import { FlexLayoutModule } from '@angular/flex-layout';
import { SharedModule } from 'src/app/shared/shared.module';
import { MatTableModule } from '@angular/material/table';
import { MaterialElevationDirective } from './material-elevation-directive';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatIconModule } from '@angular/material/icon';
import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [IndexComponent, MaterialElevationDirective],
  imports: [
    CommonModule,
    SharedModule,
    IndexRoutingModule,
    MatButtonModule,
    MatCardModule,
    MatDividerModule,
    MatTableModule,
    MatSelectModule,
    FlexLayoutModule,
    FormsModule,
    MatTooltipModule,
    MatIconModule,
  ],
  providers: [],
})
export class IndexModule {}

import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DocscRoutingModule } from './docsc-routing.module';
import { DocscComponent } from './docsc.component';


@NgModule({
  declarations: [DocscComponent],
  imports: [
    CommonModule,
    DocscRoutingModule
  ]
})
export class DocscModule { }

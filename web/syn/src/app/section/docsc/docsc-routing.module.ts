import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { DocscComponent } from './docsc.component';

const routes: Routes = [
  {
    path: '',
    component: DocscComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class DocscRoutingModule { }

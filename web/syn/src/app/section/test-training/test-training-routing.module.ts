import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { TestTrainingComponent } from './test-training.component';

const routes: Routes = [
  {
    path: '',
    component: TestTrainingComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class TestTrainingRoutingModule { }

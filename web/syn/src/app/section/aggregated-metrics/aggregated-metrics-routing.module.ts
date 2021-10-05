import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { AggregatedMetricsComponent } from './aggregated-metrics.component';

const routes: Routes = [
  {
    path: '',
    component: AggregatedMetricsComponent,
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class AggregatedMetricsRoutingModule {}

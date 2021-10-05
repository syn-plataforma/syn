import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { DocscComponent } from './section/docsc/docsc.component';
import { IndexComponent } from './section/index/index.component';
import { TestTrainingComponent } from './section/test-training/test-training.component';

const routes: Routes = [
  {
    path: '',
    redirectTo: 'home',
    pathMatch: 'full',
  },
  {
    path: '',
    loadChildren: () =>
      import('./section/index/index.module').then((m) => m.IndexModule),
  },
  {
    path: 'docs',
    loadChildren: () =>
      import('./section/docsc/docsc.module').then((m) => m.DocscModule),
  },
  {
    path: 'test',
    loadChildren: () =>
      import('./section/test-training/test-training.module').then(
        (m) => m.TestTrainingModule
      ),
  },
  {
    path: 'metrics',
    loadChildren: () =>
      import('./section/metrics/metrics.module').then((m) => m.MetricsModule),
  },
  {
    path: 'dashboard',
    loadChildren: () =>
      import('./section/dashboard/dashboard.module').then(
        (m) => m.DashboardModule
      ),
  },
  {
    path: 'aggregated-metrics',
    loadChildren: () =>
      import('./section/aggregated-metrics/aggregated-metrics.module').then(
        (m) => m.AggregatedMetricsModule
      ),
  },
  {
    path: '',
    loadChildren: () =>
      import('./shared/shared.module').then((m) => m.SharedModule),
  },

  {
    path: '**',
    redirectTo: '404',
  },
];

@NgModule({
  imports: [
    RouterModule.forRoot(routes, { scrollPositionRestoration: 'enabled' }),
  ],
  exports: [RouterModule],
})
export class AppRoutingModule {}

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AggregatedMetricsComponent } from './aggregated-metrics.component';

describe('AggregatedMetricsComponent', () => {
  let component: AggregatedMetricsComponent;
  let fixture: ComponentFixture<AggregatedMetricsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AggregatedMetricsComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(AggregatedMetricsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

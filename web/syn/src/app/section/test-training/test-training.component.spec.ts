import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TestTrainingComponent } from './test-training.component';

describe('TestTrainingComponent', () => {
  let component: TestTrainingComponent;
  let fixture: ComponentFixture<TestTrainingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TestTrainingComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TestTrainingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

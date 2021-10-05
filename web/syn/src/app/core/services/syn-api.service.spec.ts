import { TestBed } from '@angular/core/testing';

import { SynApiService } from './syn-api.service';

describe('SynApiService', () => {
  let service: SynApiService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SynApiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

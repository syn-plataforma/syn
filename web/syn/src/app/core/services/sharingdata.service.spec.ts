import { TestBed } from '@angular/core/testing';

import { SharingdataService } from './sharingdata.service';

describe('SharingdataService', () => {
  let service: SharingdataService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SharingdataService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class SharingdataService {
  private data: any = '';

  private table;

  private allStepParamsRequest;

  private requestData;

  constructor() {}

  setData(data: any) {
    this.data = data;
  }

  getData(): any {
    return this.data;
  }

  setTable(table: any) {
    this.table = table;
  }

  getTable() {
    return this.table;
  }

  setAllStepParamsRequest(allStepParamsRequest: any) {
    this.allStepParamsRequest = allStepParamsRequest;
  }

  getAllStepParamsRequest() {
    return this.allStepParamsRequest;
  }

  setRequestData(requestData: any) {
    this.requestData = requestData;
  }

  getRequestData() {
    return this.requestData;
  }
}

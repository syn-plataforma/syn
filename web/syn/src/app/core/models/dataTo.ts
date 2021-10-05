/* eslint-disable no-underscore-dangle */

import { IncidenciaRegistrada } from './incidenciaRegistrada';
import { NlpRequest } from './nlp';
import { VectorizerRequest, VectorizerResponse } from './vectorizer';

/**
 * Clase que sirva para formar un objeto para pasar datos al presenter {@Link NlpComponent} en TourModule
 *
 */
export class DataToNlpModel {
  constructor(public old: string, public response: Array<string>) {}
}

/**
 * Clase que sirva para formar un objeto para pasar datos al presenter {@Link EmbbedingComponent} en {@Link TourModule}
 *
 */
export class DataToEmbeddingModel {
  private _data: Array<DataToEmbeddingModelRow> = [];
  constructor(
    private old: Array<string>,
    private response: Array<Array<number>>
  ) {
    for (let i = 0; i < old.length; i++) {
      const newDataToEmbeddingModelRow = new DataToEmbeddingModelRow(
        this.old[i],
        this.response[i]
      );
      this._data.push(newDataToEmbeddingModelRow);
    }
  }

  public get data(): Array<DataToEmbeddingModelRow> {
    return this._data;
  }

  public set data(v: Array<DataToEmbeddingModelRow>) {
    this._data = v;
  }
}

/**
 * Clase que sirva para formar una fila de datos de {@Link DataToEmbeddingModel}
 */
export class DataToEmbeddingModelRow {
  constructor(public old: string, public vector: Array<number>) {}
}

/**
 * Clase que sirva para formar un objeto para pasar datos al presenter {@Link CodebookComponent} en {@Link TourModule}
 */
export class DataToCodebookModel {
  private _data: Array<DataToCodebookModelRow> = [];
  constructor(
    public old: Array<Array<number>>,
    public response: Array<number>,
    public words: Array<string>
  ) {
    for (let i = 0; i < old.length; i++) {
      const newDataToCodebookModelRow = new DataToCodebookModelRow(
        this.old[i],
        this.response[i],
        this.words[i]
      );
      this._data.push(newDataToCodebookModelRow);
    }
  }

  public get data(): Array<DataToCodebookModelRow> {
    return this._data;
  }

  public set data(v: Array<DataToCodebookModelRow>) {
    this._data = v;
  }
}

/**
 * Clase que sirva para formar una fila de datos de {@Link DataToCodebookModel}
 */
export class DataToCodebookModelRow {
  constructor(
    public old: Array<number>,
    public response: number,
    public word: string
  ) {}
}

/**
 * Clase que sirva para formar un objeto para pasar datos al presenter {@Link VectorizerComponent} en {@Link TourModule}
 */
export class DataToVectorizerModel {
  public _data: Array<DataToVectorizerModelRow> = [];
  constructor(
    public old?: VectorizerRequest,
    public response?: VectorizerResponse
  ) {
    this._data.push(
      new DataToVectorizerModelRow(
        'INCIDENCE_PRODUCT',
        'ONEHOT_PRODUCT',
        old.product,
        response.result.priority_ohe
      )
    );

    this._data.push(
      new DataToVectorizerModelRow(
        'ASSOCIATED_CODEBOOK',
        'TF_IDF',
        old.description,
        response.result.description
      )
    );

    this._data.push(
      new DataToVectorizerModelRow(
        'INCIDENCE_SEVERITY',
        'ONEHOT_SEVERITY',
        old.bugSeverity,
        response.result.bug_severity_ohe
      )
    );

    this._data.push(
      new DataToVectorizerModelRow(
        'INCIDENCE_PRIORITY',
        'ONEHOT_PRIORITY',
        old.priority,
        response.result.priority_le
      )
    );

    this._data.push(
      new DataToVectorizerModelRow(
        'INCIDENCE_COMPONENT',
        'ONEHOT_COMPONENT',
        old.component,
        response.result.component_ohe
      )
    );
  }

  public get data(): Array<DataToVectorizerModelRow> {
    return this._data;
  }

  public set data(v: Array<DataToVectorizerModelRow>) {
    this._data = v;
  }
}

/**
 * Clase que sirva para formar una fila de datos de {@Link DataToVectorizerModel}
 */
export class DataToVectorizerModelRow {
  constructor(
    public oldText: any,
    public responseText: any,
    public old: any,
    public response: any
  ) {}
}

export class DataToFinalStep {
  constructor(
    public old?: NlpRequest,
    public response?: any,
    public treatment?: string,
    public classification?: string,
    public priorization?: string
  ) {}
}

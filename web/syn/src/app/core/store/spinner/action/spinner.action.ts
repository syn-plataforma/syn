/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable no-shadow */
import { createAction } from '@ngrx/store';

export enum SpinnerActionTypes {
  ShowSpinner = '[Spinner] Show Spinner',
  HideSpinner = '[Spinner] Hide Spinner',
}

export const ShowSpinner = createAction(SpinnerActionTypes.ShowSpinner);

export const HideSpinner = createAction(SpinnerActionTypes.HideSpinner);

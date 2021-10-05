/* eslint-disable prefer-arrow/prefer-arrow-functions */
/* eslint-disable @typescript-eslint/naming-convention */
import { HideSpinner, ShowSpinner } from '../action/spinner.action';
import { Action, createReducer, on } from '@ngrx/store';
import { InitialSpinnerState, Spinner } from '../model/spinner';

const SpinnerReducer = createReducer(
  { ...InitialSpinnerState },
  on(ShowSpinner, onShowSpinner),
  on(HideSpinner, onHideSpinner)
);

export function reducer(state: Spinner | undefined, action: Action) {
  return SpinnerReducer(state, action);
}

function onShowSpinner(state: Spinner) {
  return { ...state, show: true };
}

function onHideSpinner(state: Spinner) {
  return { ...state, show: false };
}

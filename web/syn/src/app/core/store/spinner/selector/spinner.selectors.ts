import { createFeatureSelector, createSelector } from '@ngrx/store';
import { Spinner } from '../model/spinner';

export const selectSpinnerState = createFeatureSelector<Spinner>('spinner');

export const showSpinner = createSelector(selectSpinnerState, (spinner: Spinner): boolean => spinner.show);

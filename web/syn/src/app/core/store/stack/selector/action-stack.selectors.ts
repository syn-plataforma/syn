import { createFeatureSelector } from '@ngrx/store';
import { State } from '../../index';
import { ActionStackState } from '../model/action-stack';

export const actionStackState = createFeatureSelector<State, ActionStackState>('actionStack');

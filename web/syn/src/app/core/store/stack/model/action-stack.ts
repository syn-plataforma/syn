import { Action } from '@ngrx/store';

export interface ActionStackState {
  [type: string]: Action;
}

export const initialActionStack: ActionStackState = {};

export const STACK_ACTION_PROPERTY = 'fromStack';

export interface CustomAction extends Action {
  [STACK_ACTION_PROPERTY]: boolean;
}

export const PARENT_ACTION_PROPERTY = 'parentAction';

export interface SonAction extends Action {
  [PARENT_ACTION_PROPERTY]: any;
}

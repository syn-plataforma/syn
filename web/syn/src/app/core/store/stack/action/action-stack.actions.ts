/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable no-shadow */
import { createAction, props } from '@ngrx/store';
import { CustomAction } from '../model/action-stack';

export enum ActionStackTypes {
  RegisterActionOnStack = '[Token] Register Action on stack',
  RemoveActionFromStack = '[Token] Remove Action from stack',
  ClearActionStack = '[Token] Clear Action Stack',
}

export const RegisterActionOnStack = createAction(
  ActionStackTypes.RegisterActionOnStack,
  props<{ action: CustomAction }>()
);

export const RemoveActionFromStack = createAction(
  ActionStackTypes.RemoveActionFromStack,
  props<{ actionType: string }>()
);

export const ClearActionStack = createAction(ActionStackTypes.ClearActionStack);

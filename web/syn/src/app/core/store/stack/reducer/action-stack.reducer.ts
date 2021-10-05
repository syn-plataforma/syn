/* eslint-disable prefer-arrow/prefer-arrow-functions */
/* eslint-disable @typescript-eslint/naming-convention */
import { Action, createReducer, on } from '@ngrx/store';
import {
  initialActionStack,
  ActionStackState,
  CustomAction,
} from '../model/action-stack';
import {
  ClearActionStack,
  RegisterActionOnStack,
  RemoveActionFromStack,
} from '../action/action-stack.actions';

const ActionStackReducer = createReducer(
  { ...initialActionStack },
  on(RegisterActionOnStack, onRegisterActionOnStack),
  on(RemoveActionFromStack, onRemoveActionFromStack),
  on(ClearActionStack, onClearActionStack)
);

export function reducer(state: ActionStackState | undefined, action: Action) {
  return ActionStackReducer(state, action);
}

function onRegisterActionOnStack(
  state: ActionStackState,
  props: { action: CustomAction }
) {
  return {
    ...state,
    [props.action.type]: {
      ...props.action,
      fromStack: true,
    },
  };
}

function onRemoveActionFromStack(
  state: ActionStackState,
  props: { actionType: string }
) {
  const newState = { ...state };
  if (newState.hasOwnProperty(props.actionType)) {
    delete newState[props.actionType];
  }

  return {
    ...newState,
  };
}

function onClearActionStack() {
  return { ...initialActionStack };
}

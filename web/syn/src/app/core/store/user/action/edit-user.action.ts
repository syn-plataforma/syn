/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable no-shadow */
import { createAction } from '@ngrx/store';
import {
  STACK_ACTION_PROPERTY,
  PARENT_ACTION_PROPERTY,
} from '../../stack/model/action-stack';

export enum EditUserActionTypes {
  UpdateUserPayload = '[User] Update User Payload',
  UpdateUserOptions = '[User] Update User Options',
  UpdateUserOptionsSuccess = '[User] Update User Options Success',
  UpdateUserOptionsFail = '[User] Update User Options Fail',
}

export const UpdateUserPayload = createAction(
  EditUserActionTypes.UpdateUserPayload,
  (data: { property: string; value: string }, fromStack = false) => ({
    ...data,
    [STACK_ACTION_PROPERTY]: fromStack,
  })
);

export const UpdateUserOptions = createAction(
  EditUserActionTypes.UpdateUserOptions,
  (fromStack = false) => ({
    [STACK_ACTION_PROPERTY]: fromStack,
  })
);

export const UpdateUserOptionsSuccess = createAction(
  EditUserActionTypes.UpdateUserOptionsSuccess,
  (
    data: { payload: string },
    parentAction = EditUserActionTypes.UpdateUserOptions
  ) => ({
    ...data,
    [PARENT_ACTION_PROPERTY]: parentAction,
  })
);
export const UpdateUserOptionsFail = createAction(
  EditUserActionTypes.UpdateUserOptionsFail,
  (
    data: { payload: string },
    parentAction = EditUserActionTypes.UpdateUserOptions
  ) => ({
    ...data,
    [PARENT_ACTION_PROPERTY]: parentAction,
  })
);

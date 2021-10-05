/* eslint-disable no-shadow */
/* eslint-disable @typescript-eslint/naming-convention */
import { createAction, props } from '@ngrx/store';
import { RefreshTokenResponse } from '../model/user';
import { ActionStackState } from '../../stack/model/action-stack';

export enum RefreshTokenActionTypes {
  RefreshToken = '[Token] Refresh Token',
  RefreshTokenSuccess = '[Token] Refresh Token Success',
  RefreshTokenFail = '[Token] Refresh Token Fail',
}

export const RefreshToken = createAction(RefreshTokenActionTypes.RefreshToken);

export const RefreshTokenSuccess = createAction(
  RefreshTokenActionTypes.RefreshTokenSuccess,
  props<{ res: RefreshTokenResponse; actions: ActionStackState }>()
);

export const RefreshTokenFail = createAction(
  RefreshTokenActionTypes.RefreshTokenFail,
  props<{ error: any }>()
);

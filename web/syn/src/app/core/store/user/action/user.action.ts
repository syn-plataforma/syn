/* eslint-disable no-shadow */
/* eslint-disable @typescript-eslint/naming-convention */
import { createAction, props } from '@ngrx/store';
import { LoginData } from '../model/login';
import {
  PARENT_ACTION_PROPERTY,
  STACK_ACTION_PROPERTY,
} from '../../stack/model/action-stack';
import { UserResponse } from 'src/app/core/api/model/user.model-response';

export enum UserActionTypes {
  LoadLoggedUser = '[User] Load Logged User from local storage',
  LoadLoggedUserSuccess = '[User] Load Logged User from local storage Success',
  LoadLoggedUserFail = '[User] Load Logged User from local storage Fail',

  LoadClaveForm = '[User] Load Clave Form',
  LoadClaveFormSuccess = '[User] Load Clave Form Success',
  LoadClaveFormFail = '[User] Load Clave Form Fail',

  SetLogin = '[User] Set Login Data',
  SetLoginSuccess = '[User] Set Login Data Success',
  LoginFail = '[User] Login Fail',

  Logout = '[User] Logout User',
  LogoutSuccess = '[User] Logout User Success',
  LogoutFail = '[User] Logout User Fail',

  LogoutOnlyFront = '[User] Logout User Only Front',
  LogoutOnlyFrontSuccess = '[User] Logout User Only Front Success',
  LogoutOnlyFrontFail = '[User] Logout User Only Front Fail',

  SetRedirectUrl = '[User] User Redirect Url',
  SetRedirectUrlSuccess = '[User] User Redirect Url Success',
}

/**
 * NOTA:
 * - Las acciones '...Success' y '...Fail' asociadas a las acciones que generan llamadas a la API, tienen un campo
 *   para guardar a qué acción hacen referencia(constante PARENT_ACTION_PROPERTY).
 * - Con este campo se centraliza el borrado del Stack de acciones, cuando se lanza una acción que tenga dicho parámetro,
 *   se lanza el evento RemoveActionFromStack con el valor recuperado.
 */

export const LoadLoggedUser = createAction(
  UserActionTypes.LoadLoggedUser,
  (fromStack = false) => ({
    [STACK_ACTION_PROPERTY]: fromStack,
  })
);

export const LoadLoggedUserSuccess = createAction(
  UserActionTypes.LoadLoggedUserSuccess,
  (
    data: { payload: UserResponse },
    parentAction = UserActionTypes.LoadLoggedUser
  ) => ({
    ...data,
    [PARENT_ACTION_PROPERTY]: parentAction,
  })
);

export const LoadLoggedUserFail = createAction(
  UserActionTypes.LoadLoggedUserFail,
  (
    data: { payload: string },
    parentAction = UserActionTypes.LoadLoggedUser
  ) => ({
    ...data,
    [PARENT_ACTION_PROPERTY]: parentAction,
  })
);

export const LoadClaveForm = createAction(
  UserActionTypes.LoadClaveForm,
  (fromStack = false) => ({
    [STACK_ACTION_PROPERTY]: fromStack,
  })
);

export const LoadClaveFormSuccess = createAction(
  UserActionTypes.LoadClaveFormSuccess,
  (
    data: { payload: string },
    parentAction = UserActionTypes.LoadClaveForm
  ) => ({
    ...data,
    [PARENT_ACTION_PROPERTY]: parentAction,
  })
);

export const LoadClaveFormFail = createAction(
  UserActionTypes.LoadClaveFormFail,
  (
    data: { payload: string },
    parentAction = UserActionTypes.LoadClaveForm
  ) => ({
    ...data,
    [PARENT_ACTION_PROPERTY]: parentAction,
  })
);

export const SetLogin = createAction(
  UserActionTypes.SetLogin,
  props<LoginData>()
);

export const SetLoginSuccess = createAction(UserActionTypes.SetLoginSuccess);

export const LoginFail = createAction(
  UserActionTypes.LoginFail,
  props<{ payload: string | undefined }>()
);

export const Logout = createAction(UserActionTypes.Logout);

export const LogoutSuccess = createAction(UserActionTypes.LogoutSuccess);

export const LogoutFail = createAction(
  UserActionTypes.LogoutFail,
  props<{ error: string }>()
);

export const LogoutOnlyFront = createAction(UserActionTypes.LogoutOnlyFront);

export const LogoutOnlyFrontSuccess = createAction(
  UserActionTypes.LogoutOnlyFrontSuccess
);

export const LogoutOnlyFrontFail = createAction(
  UserActionTypes.LogoutOnlyFrontFail,
  props<{ error: string }>()
);

export const SetRedirectUrl = createAction(
  UserActionTypes.SetRedirectUrl,
  props<{ payload: string }>()
);

export const SetRedirectUrlSuccess = createAction(
  UserActionTypes.SetRedirectUrlSuccess,
  props<{ url: string }>()
);

// NOTE: Para crear grupos de tipos que pueden ser usados en los effects.
// const allUserActions = union({LoadLoggedUser, LoadAppearanceClaveForm, LoadAppearanceClaveFormSuccess, LoadAppearanceClaveFormFail});
// export type UserActionsUnion = typeof allUserActions;

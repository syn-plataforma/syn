/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable prefer-arrow/prefer-arrow-functions */
import {
  DEFAULT_LOGIN_REDIRECT_URL,
  initialUser,
  RefreshTokenResponse,
  User,
} from '../model/user';
import {
  LoadClaveForm,
  LoadClaveFormFail,
  LoadClaveFormSuccess,
  LoadLoggedUser,
  LoginFail,
  Logout,
  LogoutFail,
  LogoutOnlyFront,
  LogoutOnlyFrontFail,
  LogoutOnlyFrontSuccess,
  LogoutSuccess,
  SetLogin,
  SetRedirectUrl,
} from '../action/user.action';
import { Action, createReducer, on } from '@ngrx/store';
import { LoginData } from '../model/login';
import { LocalStorage } from '../../local-storage';
import { initialPerson } from '../model/person';
import { LoggerService } from 'src/app/core/services/logger.service';
import {
  RefreshToken,
  RefreshTokenFail,
  RefreshTokenSuccess,
} from '../action/token.action';
import { ActionStackState } from '../../stack/model/action-stack';
import {
  UpdateUserPayload,
  UpdateUserOptions,
  UpdateUserOptionsSuccess,
  UpdateUserOptionsFail,
} from '../action/edit-user.action';

const UserReducer = createReducer(
  { ...initialUser },
  on(LoadClaveForm, onLoadClaveForm),
  on(LoadClaveFormSuccess, onLoadClaveFormSuccess),
  on(LoadLoggedUser, onLoadLoggedUser),
  on(LoginFail, onLoginFail),
  on(SetLogin, onSetLogin),
  on(LoadClaveFormFail, onLoadClaveFormFail),
  on(Logout, onLogout),
  on(LogoutSuccess, onLogoutSuccess),
  on(LogoutFail, onLogoutFail),
  on(LogoutOnlyFront, onLogout),
  on(LogoutOnlyFrontSuccess, onLogoutSuccess),
  on(LogoutOnlyFrontFail, onLogoutFail),
  on(SetRedirectUrl, onSetRedirectUrl),
  on(RefreshToken, onRefreshToken),
  on(RefreshTokenSuccess, onRefreshTokenSuccess),
  on(RefreshTokenFail, onRefreshTokenFail),
  on(UpdateUserPayload, onUpdateUserPayload),
  on(UpdateUserOptions, onUpdateUserOptions),
  on(UpdateUserOptionsSuccess, onUpdateUserOptionsSuccess),
  on(UpdateUserOptionsFail, onUpdateUserOptionsFail)
);

export function reducer(state: User | undefined, action: Action) {
  return UserReducer(state, action);
}

function onLoadClaveForm(state: User) {
  return {
    ...state,
  };
}

function onLoadClaveFormSuccess(state: User, data: { payload: string }) {
  return {
    ...state,
    claveForm: data.payload,
  };
}

function onLoadLoggedUser(state: User) {
  return {
    ...initialUser,
    person: LocalStorage.select('person', { ...initialPerson }),
    token: LocalStorage.select('token', ''),
    refreshToken: LocalStorage.select('refreshToken', ''),
    refreshTokenExpiration: LocalStorage.select('refreshTokenExpiration', ''),
    redirectUrl: LocalStorage.select('redirectUrl', DEFAULT_LOGIN_REDIRECT_URL),
    claveForm: '',
    authenticating: false,
    logged: true,
  };
}

function onLoginFail(state: User, data: { payload: string | undefined }) {
  return {
    ...state,
    error: data.payload,
    authenticating: false,
    logged: false,
  };
}

function onSetLogin(state: User, data: LoginData) {
  LocalStorage.set('person', data.person);
  LocalStorage.set('token', data.token);
  LocalStorage.set('refreshToken', data.refreshToken);
  LocalStorage.set('refreshTokenExpiration', data.refreshTokenExpiration);

  return {
    ...state,
    ...data,
    authenticating: false,
    logged: true,
  };
}

function onLoadClaveFormFail(state: User, data: { payload: string }) {
  return { ...initialUser };
}

function onLogout(state: User) {
  return {
    ...state,
    logoutInProgress: true,
  };
}

function onLogoutSuccess(state: User) {
  LocalStorage.clear();

  return { ...initialUser };
}

function onLogoutFail(state: User, data: { error: string }) {
  LocalStorage.clear();

  return { ...initialUser };
}

function onSetRedirectUrl(state: User, data: { payload: string }) {
  LocalStorage.set('redirectUrl', data.payload);

  return {
    ...state,
    redirectUrl: data.payload,
  };
}

function onRefreshToken(state: User) {
  return {
    ...state,
    refreshingToken: true,
  };
}

function onRefreshTokenSuccess(
  state: User,
  props: { res: RefreshTokenResponse; actions: ActionStackState }
) {
  LocalStorage.set('token', props.res.token);
  LocalStorage.set('refreshToken', props.res.refreshToken);
  LocalStorage.set('refreshTokenExpiration', props.res.refreshTokenExpiration);
  LoggerService.log('onRefreshTokenSuccess', props.res);

  return {
    ...state,
    token: props.res.token,
    refreshToken: props.res.refreshToken,
    refreshingToken: false,
    refreshTokenExpiration: props.res.refreshTokenExpiration,
  };
}

function onRefreshTokenFail(state: User) {
  LoggerService.log('onRefreshTokenFail');
  return {
    ...state,
    refreshToken: '',
    refreshingToken: false,
  };
}

function onUpdateUserPayload(
  state: User,
  data: { property: string; value: string }
) {
  return {
    ...state,
    [data.property]: data.value,
  };
}

function onUpdateUserOptions(state: User) {
  return {
    ...state,
  };
}

function onUpdateUserOptionsSuccess(state: User, data: { payload: string }) {
  return {
    ...state,
  };
}

function onUpdateUserOptionsFail(state: User, data: { payload: string }) {
  return {
    ...state,
  };
}

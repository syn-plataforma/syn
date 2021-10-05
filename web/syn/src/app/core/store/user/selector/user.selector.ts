import { createFeatureSelector, createSelector } from '@ngrx/store';
import { State } from '../../index';
import { User } from '../model/user';
import { Person } from '../model/person';
import { LoggerService } from '../../../service/logger.service';

export const selectUserState = createFeatureSelector<State, User>('user');

export const getPerson = createSelector(selectUserState, (user: User): Person => user.person);
export const getClaveForm = createSelector(selectUserState, (user: User): string => user.claveForm);
export const getAppearanceClaveForm = createSelector(selectUserState, (user: User): string => user.appearanceClaveForm);
export const getRedirectUrl = createSelector(selectUserState, (user: User): string => user.redirectUrl);
export const isLogged = createSelector(getPerson, (person: Person): boolean => !!person.identifier);
export const getError = createSelector(selectUserState, (user: User): string => user.error);
export const getToken = createSelector(selectUserState, (user: User): string => user.token);
export const getRefreshToken = createSelector(selectUserState, (user: User): string => {
  LoggerService.log('Get refreshToken', user.refreshToken);
  return user.refreshToken;
});
export const getRefreshTokenExpiration = createSelector(selectUserState, (user: User): number => {
  LoggerService.log('Get refreshTokenExpiration', user.refreshTokenExpiration);
  return user.refreshTokenExpiration;
});

export const isValid = createSelector(selectUserState, isLogged, (user: User): boolean => {

  const tokenExpiration = user.refreshTokenExpiration;

  if (isLogged && tokenExpiration) {

    return tokenExpiration > new Date().getTime();
  }

  return false;

});

export const isLogoutInProgress = createSelector(selectUserState, (user: User): boolean => user.logoutInProgress);
export const isRefreshingTokenInProgress = createSelector(selectUserState, (user: User): boolean => user.refreshingToken);

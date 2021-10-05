import { initialPerson, Person } from './person';
import { RoutesEnum } from 'src/app/core/enum/route.enum';

export const DEFAULT_LOGIN_REDIRECT_URL = RoutesEnum.HOME;

export interface User {
  person: Person;
  token: string;
  refreshToken: string;
  refreshingToken: boolean;
  refreshTokenExpiration: number;
  claveForm: string;
  appearanceClaveForm: string;
  redirectUrl: string;
  error: string;
  authenticating: boolean;
  logged: boolean;
  logoutInProgress: boolean;
  language: string;
  alerts: boolean;
}

export const initialUser: User = {
  person: { ...initialPerson },
  token: '',
  refreshToken: '',
  refreshingToken: false,
  refreshTokenExpiration: null,
  claveForm: '',
  appearanceClaveForm: '',
  redirectUrl: DEFAULT_LOGIN_REDIRECT_URL,
  error: '',
  authenticating: false,
  logged: false,
  logoutInProgress: false,
  language: '',
  alerts: false,
};

export interface RefreshTokenResponse {
  token: string;
  refreshToken: string;
  refreshTokenExpiration: number;
}

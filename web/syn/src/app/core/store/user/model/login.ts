import { Person } from './person';

export interface LoginData {
  person: Person;
  token: string;
  refreshToken: string;
  refreshTokenExpiration: number;
}

/* eslint-disable max-len */
/* eslint-disable @typescript-eslint/naming-convention */
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Router } from '@angular/router';
import { Person } from '../store/user/model/person';
import { environment } from '../../../environments/environment';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { LoginRequest, LoginResponse } from '../models/login';
import { THIS_EXPR } from '@angular/compiler/src/output/output_ast';

const httpOptions = {
  headers: new HttpHeaders({
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'application/json',
  }),
};
@Injectable({
  providedIn: 'root',
})
export class AuthService {
  redirectUrl: Observable<string>;
  claveForm: Observable<string>;
  isLogged: Observable<boolean>;
  person: Observable<Person>;
  token: Observable<string>;
  refreshToken$: Observable<string>;
  refreshTokenExpiration$: Observable<number>;

  constructor(private http: HttpClient) {}

  login(loginRequest: LoginRequest) {
    this.http
      .post<LoginResponse>(
        environment.loginUri + '/',
        loginRequest.transform(),
        httpOptions
      )
      .subscribe(
        (result) => {
          this.setToken(result.access_token);
        },
        (error) => {
          console.log(error);
        }
      );
  }

  setToken(token: string) {
    localStorage.setItem('token', token);
  }

  getToken() {
    localStorage.getItem('token');
  }
}

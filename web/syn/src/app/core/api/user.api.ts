import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';
import { HttpClient } from '@angular/common/http';
import { Store } from '@ngrx/store';
import { State } from '../store';
import { HideSpinner, ShowSpinner } from '../store/spinner/action/spinner.action';
import { map } from 'rxjs/operators';
import { UserResponse } from './model/user.model-response';

@Injectable ({
    providedIn: 'root'
})

export class UserApi {

    private userApiBase = environment.apiUri;

    constructor(
        private http: HttpClient,
        private store: Store<State>
    ) { }

    getUser$() {
        this.store.dispatch(ShowSpinner());
        return this.http.get(`${this.userApiBase}/users/2`).pipe(map( res => {
            this.store.dispatch(HideSpinner());
            return res as UserResponse;
        }));
    }

}

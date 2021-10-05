import { Injectable } from '@angular/core';
import { Store } from '@ngrx/store';
import { LoadLoggedUser } from '../action/user.action';
import { State } from '../../index';
import { UpdateUserPayload, UpdateUserOptions } from '../action/edit-user.action';

@Injectable({
  providedIn: 'root'
})

export class UserFacade {

  constructor(private store: Store<State>) {}

  getLoggedUser = () => this.store.dispatch(LoadLoggedUser());
  updatePayload = (payload) => this.store.dispatch(UpdateUserPayload(payload));
  updateUserOptions = () => this.store.dispatch(UpdateUserOptions());
  // INIT SELECTOR
    // getUser = () => this.store.dispatch(getUser());
  // END SELECTORS
}

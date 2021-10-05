import { Injectable } from '@angular/core';
import { Store } from '@ngrx/store';
import { showSpinner } from '../selector/spinner.selectors';
import { ShowSpinner, HideSpinner } from '../action/spinner.action';
import { State } from '../../index';

@Injectable({
  providedIn: 'root'
})

export class SpinnerFacade {

  constructor(private store: Store<State>) {}

  showSpinner = () => this.store.dispatch(ShowSpinner());
  hideSpinner = () => this.store.dispatch(HideSpinner());

  // INIT SELECTOR
  getShowSpinner = () => this.store.select(showSpinner);
  // END SELECTORS
}

import {
    ActionReducerMap,
    MetaReducer,
  } from '@ngrx/store';
import { environment } from '../../../environments/environment';
import { Spinner } from './spinner/model/spinner';
import { reducer as ActionStackReducer } from './stack/reducer/action-stack.reducer';
import { reducer as SpinnerReducer } from './spinner/reducer/spinner.reducers';
import { reducer as UserReducer } from './user/reducer/user.reducer';
import { ActionStackState } from './stack/model/action-stack';
import { User } from './user/model/user';

export interface State {
    actionStack: ActionStackState;
    spinner: Spinner;
    user: User;
  }

export const reducers: ActionReducerMap<State> = {
    actionStack: ActionStackReducer,
    spinner: SpinnerReducer,
    user: UserReducer,
  };

export const metaReducers: MetaReducer<State>[] = !environment.production ? [] : [];

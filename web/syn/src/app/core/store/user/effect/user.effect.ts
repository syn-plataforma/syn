import { Injectable } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import {
  UserActionTypes,
  LoadLoggedUserSuccess,
  LoadLoggedUserFail,
} from '../action/user.action';
import { catchError, map, mergeMap } from 'rxjs/operators';
import { of } from 'rxjs';
import { LoggerService } from 'src/app/core/services/logger.service';
import { UserApi } from 'src/app/core/api/user.api';
import { UserResponse } from 'src/app/core/api/model/user.model-response';

@Injectable()
export class UserEffects {
  loadLoggedUser = createEffect(() =>
    this.actions$.pipe(
      ofType(UserActionTypes.LoadLoggedUser),
      mergeMap(() =>
        this.api.getUser$().pipe(
          map((user: UserResponse) =>
            LoadLoggedUserSuccess({ payload: user as UserResponse })
          ),
          catchError((err) => of(LoadLoggedUserFail({ payload: err.message })))
        )
      )
    )
  );
  constructor(
    private actions$: Actions,
    private api: UserApi,
    private logger: LoggerService
  ) {}
}

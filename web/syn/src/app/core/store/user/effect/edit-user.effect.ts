import { Injectable } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import { UserApi } from 'src/app/core/api/user.api';
import { LoggerService } from 'src/app/core/services/logger.service';
import {
  EditUserActionTypes,
  UpdateUserOptionsSuccess,
  UpdateUserOptionsFail,
} from '../action/edit-user.action';
import { mergeMap, map, catchError } from 'rxjs/operators';
import { UserResponse } from 'src/app/core/api/model/user.model-response';
import { of } from 'rxjs';

@Injectable()
export class EditUserEffects {
  updateUserOptions = createEffect(() =>
    this.actions$.pipe(
      ofType(EditUserActionTypes.UpdateUserOptions),
      mergeMap(() =>
        this.api.getUser$().pipe(
          map((user: UserResponse) =>
            UpdateUserOptionsSuccess({ payload: 'ok' })
          ),
          catchError((err) =>
            of(UpdateUserOptionsFail({ payload: err.message }))
          )
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

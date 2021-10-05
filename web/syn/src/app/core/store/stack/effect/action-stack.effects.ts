import { Injectable } from '@angular/core';
import { Actions, createEffect } from '@ngrx/effects';
import { filter, mergeMap } from 'rxjs/operators';
import { of } from 'rxjs';
import { Action } from '@ngrx/store';
import {
  RegisterActionOnStack,
  RemoveActionFromStack,
} from '../action/action-stack.actions';
import {
  CustomAction,
  PARENT_ACTION_PROPERTY,
  SonAction,
  STACK_ACTION_PROPERTY,
} from '../model/action-stack';

@Injectable()
export class ActionStackEffects {
  registerActionOnStack$ = createEffect(() =>
    this.actions$.pipe(
      filter((action: CustomAction) =>
        ActionStackEffects.isStackableActionType(action)
      ),
      mergeMap((action: CustomAction) => of(RegisterActionOnStack({ action })))
    )
  );

  removeActionFromStack$ = createEffect(() =>
    this.actions$.pipe(
      filter((action: SonAction) => ActionStackEffects.isSonActionType(action)),
      mergeMap((action: SonAction) =>
        of(
          RemoveActionFromStack({ actionType: action[PARENT_ACTION_PROPERTY] })
        )
      )
    )
  );
  constructor(private actions$: Actions) {}

  static isRegisteredType(action: Action): boolean {
    return Object.values(RegistryActionTypes).includes(action.type);
  }

  static isStackableActionType(action: CustomAction): boolean {
    return (
      ActionStackEffects.isRegisteredType(action) &&
      action.hasOwnProperty(STACK_ACTION_PROPERTY) &&
      action[STACK_ACTION_PROPERTY] === false
    );
  }

  static isSonActionType(action: Action): boolean {
    return (
      ActionStackEffects.isRegisteredType(action) &&
      action.hasOwnProperty(PARENT_ACTION_PROPERTY) &&
      action[PARENT_ACTION_PROPERTY]
    );
  }
}

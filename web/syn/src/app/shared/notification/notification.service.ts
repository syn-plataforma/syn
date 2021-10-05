/* eslint-disable @typescript-eslint/member-ordering */
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class NotificationService {
  private showBehaviorSubject = new BehaviorSubject<boolean>(false);
  public show$ = this.showBehaviorSubject.asObservable();
  private typeBehaviorSubject = new BehaviorSubject<
    'spinner' | 'info' | 'warning' | 'error' | 'confirm'
  >('info');
  public type$ = this.typeBehaviorSubject.asObservable();
  private titleBehaviorSubject = new BehaviorSubject<string>('');
  public title$ = this.titleBehaviorSubject.asObservable();
  private descriptionBehaviorSubject = new BehaviorSubject<string>('');
  public description$ = this.descriptionBehaviorSubject.asObservable();
  private positionBehaviorSubject = new BehaviorSubject<
    'center' | 'right' | 'left'
  >('center');
  public position$ = this.positionBehaviorSubject.asObservable();
  constructor() {}

  hide = () => this.showBehaviorSubject.next(false);

  show = () => this.showBehaviorSubject.next(true);

  type = (value: 'spinner' | 'info' | 'warning' | 'error' | 'confirm') =>
    this.typeBehaviorSubject.next(value);

  title = (value: string) => this.titleBehaviorSubject.next(value);

  description = (value: string) => this.descriptionBehaviorSubject.next(value);

  center = () => this.positionBehaviorSubject.next('center');

  right = () => this.positionBehaviorSubject.next('right');

  left = () => this.positionBehaviorSubject.next('left');

  spinner = (description = '', title = '') => {
    this.show();
    this.type('spinner');
    this.positionBehaviorSubject.next('center');
    this.title(title);
    this.description(description);
  };

  spinnerLoad = (description = '') => {
    this.spinner(description, 'Cargando...');
  };

  spinnerSave = (description = '') => {
    this.spinner(description, 'Guardando...');
  };

  info = (title = '', description = 'Información') => {
    this.show();
    this.type('info');
    this.right();
    this.title(title);
    this.description(description);
  };

  warning = (description = '', title = 'Aviso') => {
    this.show();
    this.type('warning');
    this.right();
    this.title(title);
    this.description(description);
  };

  error = (description = '', title = 'Upsss!') => {
    this.typeBehaviorSubject.next('error');
    this.type('error');
    this.right();
    this.title(title);
    this.description(description);
  };

  confirm = (description = '', title = '') => {
    this.show();
    this.type('confirm');
    this.right();
    this.title(title);
    this.description(description);
  };

  reset = () => {
    this.hide();
    this.type('info');
    this.title('');
    this.description('');
    this.center();
  };
}

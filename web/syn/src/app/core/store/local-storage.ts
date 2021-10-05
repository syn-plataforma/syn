import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class LocalStorage {

  constructor() {
  }

  static check(): Storage {
    return window.localStorage;
  }

  static select(key: string, defaultValue: any = null): any {
    return window.localStorage.getItem(key)
      ? JSON.parse(window.localStorage.getItem(key))
      : defaultValue;
  }

  static set(key: string, value: any): void {
    window.localStorage.setItem(key, JSON.stringify(value));
  }

  static remove(key: string): void {
    window.localStorage.removeItem(key);
  }

  static clear(): void {
    window.localStorage.clear();
  }
}

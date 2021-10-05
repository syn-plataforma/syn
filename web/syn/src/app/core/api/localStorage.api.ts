import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class LocalStorageApi {
  constructor() {}

  select(key: string, defaultValue: any = null): any {
    return window.localStorage.getItem(key)
      ? JSON.parse(window.localStorage.getItem(key))
      : defaultValue;
  }

  set(key: string, value: any, stringify: boolean = true): void {
    window.localStorage.setItem(key, stringify ? JSON.stringify(value) : value);
  }

  setProperty(
    propertyPath: string,
    value: any,
    stringify: boolean = true
  ): void {
    const path = propertyPath.split('.');
    if (path.length === 1) {
      this.set(propertyPath, value, stringify);
    }
    let item = this.select(path[0], {});
    const itemKey = path[0];
    item = this.setPropertyRecursive(item, path.slice(1), value, stringify);
    this.set(itemKey, item);
  }
  remove(key: string): void {
    window.localStorage.removeItem(key);
  }

  clear(): void {
    window.localStorage.clear();
  }
  private setPropertyRecursive(
    item,
    path,
    value: any,
    stringify: boolean = true
  ) {
    const key = path[0];
    if (!item) {
      item = { key: undefined };
    }
    if (path.length > 1) {
      item[key] = this.setPropertyRecursive(
        item[path[0]],
        path.slice(1),
        value,
        stringify
      );
    } else if (path.length === 1) {
      item[key] = stringify ? JSON.stringify(value) : value;
    }

    return item;
  }
}

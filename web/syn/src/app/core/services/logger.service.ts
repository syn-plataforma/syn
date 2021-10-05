/* eslint-disable no-underscore-dangle */
/* eslint-disable prefer-arrow/prefer-arrow-functions */
import { Injectable } from '@angular/core';
import { ActivatedRoute, ParamMap } from '@angular/router';

let DEBUG = false;
@Injectable({
  providedIn: 'root',
})
export class LoggerService {
  private debug = false;

  constructor(private route: ActivatedRoute) {
    this.route.queryParamMap.subscribe((params: ParamMap) => {
      const queryDebug = params.get('debug') || localStorage.getItem('debug');

      this.debug = queryDebug === '1';
      DEBUG = this.debug;
      if (this.debug) {
        this.log('Debug activated');
      }
    });
  }

  static log(...params) {
    if (DEBUG) {
      console.log(_getTime(), ...params);
    }
  }

  static warn(...params) {
    if (DEBUG) {
      console.warn(_getTime(), ...params);
    }
  }

  static error(...params) {
    if (DEBUG) {
      console.error(_getTime(), ...params);
    }
  }

  log(...params) {
    if (this.debug) {
      console.log(_getTime(), ...params);
    }
  }

  warn(...params) {
    if (this.debug) {
      console.warn(_getTime(), ...params);
    }
  }

  error(...params) {
    if (this.debug) {
      console.error(_getTime(), ...params);
    }
  }
}

function _getTime(): string {
  const date = new Date();
  const hours = date.getHours() < 10 ? `0${date.getHours()}` : date.getHours();
  const minutes =
    date.getMinutes() < 10 ? `0${date.getMinutes()}` : date.getMinutes();
  const seconds =
    date.getSeconds() < 10 ? `0${date.getSeconds()}` : date.getSeconds();
  const milliseconds = date.getMilliseconds();

  return `${hours}:${minutes}:${seconds}:${milliseconds}`;
}

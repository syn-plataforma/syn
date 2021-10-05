import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';
import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';


@Injectable()
export class HttpSwitchUserInterceptor implements HttpInterceptor {
  constructor() {}
  // Request Interceptor to append x-user-key Header
  private static setXSwitchUserHeader(req: HttpRequest<any>): HttpRequest<any> {
    if (req) {
      const switchUser = localStorage.getItem('x-user-key');
      if (switchUser) {
        return req.clone({ setHeaders: { 'x-user-key': switchUser } });
      }

      return req;
    }
  }

  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    return next.handle(HttpSwitchUserInterceptor.setXSwitchUserHeader(req));
  }
}

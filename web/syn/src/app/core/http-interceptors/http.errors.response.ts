import { throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Injectable } from '@angular/core';
import { HttpErrorResponse, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';
import { Router } from '@angular/router';
// import { LoginService } from '../../login/login.service';
import { NotificationService } from '../../shared/notification/notification.service';

@Injectable()
export class HttpErrorsResponseInterceptor implements HttpInterceptor {
  constructor(
    private router: Router,
    private notification: NotificationService,
    // private loginService: LoginService
  ) {}

  intercept(req: HttpRequest<any>, next: HttpHandler): any {
    return next.handle(req.clone())
      .pipe(
        catchError((event) => {
          if (event instanceof HttpErrorResponse) {
            return this.handleError(event);
          }
        })
      );
  }

  private handleError(error: HttpErrorResponse): any {
    if (error.status === 400) {
      this.notification.reset();
      console.log('Bad request', error);
      return throwError(error);
    // } else if (error.status === 401) {
    //   console.error('Unauthorized', error);
    //   this.notification.warning('Su sesión ha caducado!');
    //   this.loginService.logout$().subscribe(() => {
    //     this.router.navigate(['/login']).finally();
    //   }).unsubscribe();
    //   return throwError(error);
    } else if (error.status === 403) {
      console.error('Forbidden', error);
      this.notification.warning('Operación no permitida!');
      return throwError(error);
    } else if (error.status === 409) {
      this.notification.reset();
      console.log('Bad request', error);
      return throwError(error);
    } else if (error.status === 500 || error.status === 405) {
      console.error('An error occurred', error);
      this.notification.error('Ocurrió un error!');
      return throwError(error);
    } else {
      return throwError(error);
    }
  }
}

import { Injectable } from '@angular/core';
import {CanActivate, ActivatedRouteSnapshot, RouterStateSnapshot, UrlTree, Router} from '@angular/router';
import { Observable } from 'rxjs';
import {AuthService} from '../services/auth.service';
import {LoggerService} from '../services/logger.service';
import {RoutesEnum} from '../enum/route.enum';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  constructor(
    private authService: AuthService,
    private router: Router,
    readonly logger: LoggerService
  ) { }
  canActivate(
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    return this.checkLogin(state.url);
  }

  canActivateChild(
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    return this.canActivate(route, state);
  }

  checkLogin(url: string): boolean {
    const person = this.authService.getUser();
    const token = this.authService.getToken();
    const refreshtoken = this.authService.getRefreshToken();
    const refreshTokenExpiration = this.authService.getRefreshTokenExpiration();

    if (this.authService.isLoggedIn()) {
      return true;
    }

    // Store the attempted URL for redirecting
    this.authService.setRedirectUrl(url);

    // Navigate to the login page with extras
    this.router.navigate([RoutesEnum.HOME], {fragment: RoutesEnum.LOGIN}).then(() => {
      this.logger.log('Go to login', person, token, refreshtoken, refreshTokenExpiration);
    });

    return false;
  }
}

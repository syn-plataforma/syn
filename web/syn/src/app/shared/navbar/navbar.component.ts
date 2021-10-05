import { Component, OnInit } from '@angular/core';
import { Router, Event, NavigationEnd } from '@angular/router';
import { timingSafeEqual } from 'crypto';
import { element } from 'protractor';

/**
 * Componente NavBar, sirve para el menú superior de todo el app.
 *
 * Se encuentra contenido dentro de {@Link AppComponent}
 */
@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.scss'],
})
export class NavbarComponent implements OnInit {
  active =
    'mat-focus-indicator mat-menu-trigger mat-button mat-button-base active';
  unactive = 'mat-focus-indicator mat-menu-trigger mat-button mat-button-base';
  isActive: boolean;

  constructor(private router: Router) {
    /**
     * Sirve para que cuando la ruta contenga '/api', el botón del navbar de API, se oscurezca como activo.
     */
    this.router.events.subscribe((event: Event) => {
      if (event instanceof NavigationEnd) {
        if (event.url.slice(0, 4) === '/api') {
          this.isActive = true;
        } else {
          this.isActive = false;
        }
      }
    });
  }

  ngOnInit(): void {}

  /**
   * Función que sirve para redirigir a la web de .
   */
  logoClick() {
    window.open('https://.es');
  }
}

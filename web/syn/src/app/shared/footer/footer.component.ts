import { Component, OnInit } from '@angular/core';

/**
 * Footer component, sirve para añadir el footer a todo el app.
 *
 * Se encuentra contenido dentro de {@Link AppComponent}
 */
@Component({
  selector: 'app-footer',
  templateUrl: './footer.component.html',
  styleUrls: ['./footer.component.scss'],
})
export class FooterComponent implements OnInit {
  constructor() {}

  ngOnInit(): void {}

  /**
   * Función que sirve para redirigir a la web de .
   */
  logoClick() {
    window.open('https://.es');
  }
}

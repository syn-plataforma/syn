import { Component, OnInit } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';
import { LoginRequest } from './core/models/login';
import { AuthService } from './core/services/auth.service';
import { environment } from 'src/environments/environment';
/**
 * Componente padre de toda el app web, si contenido directo son los siguientes componentes:
 *
 * 1. {@Link IndexComponent} que se encarga de contener toda las vistas del app.
 * 2. {@Link NavbarComponent} que es el navegador del app.
 * 3. {@Link IndexComponent} componente que contiene el "home del app"
 * 4. {@Link TestTrainingComponent} componente de "modo directo"
 * 5. {@Link TourComponent} componente del "modo guiado"
 * 6. {@Link NotFoundComponent} componente que sirve para redirigir al usuario a una pantalla de not found
 */

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent implements OnInit {
  authService: AuthService;
  title = 'frontend';

  /**
   * Gracias a este constructor podemos internacionalizar el app
   * mediante la interpolación de lo siguiente en los html: {{'PALABRA CLAVE'|translate}}.
   *
   * La palabra clave, se contiene en unos archivos json en assets/i18n
   *
   * @param translate Se crea un servicio para el i18n del app
   */
  constructor(translate: TranslateService, authService: AuthService) {
    const browserLang: string = translate.getBrowserLang();
    translate.use(browserLang.match(/en|es/) ? browserLang : 'es');
    this.authService = authService;
  }

  ngOnInit() {
    this.authService.login(
      new LoginRequest(environment.user, environment.password)
    );
  }
}

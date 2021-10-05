import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
/**
 * Componente usado para cuando el usuario pone una dirección incorrecta.
 *
 * Contenido por {@Link AppComponent}
 *
 */
@Component({
  selector: 'app-not-found',
  templateUrl: './not-found.component.html',
  styleUrls: ['./not-found.component.scss']
})
export class NotFoundComponent implements OnInit {

  constructor(private router: Router) { }

  ngOnInit(): void {
  }

  goIndex(){
    this.router.navigateByUrl('');

  }

}

import { router } from "./index.routes";
import Navbar from "./components/core/navbar/navbar";
import Footer from "./components/core/footer/footer";

import "bootstrap/dist/css/bootstrap.min.css";
import "./assets/sass/main.scss";

/**
 * Este método se ejecuta al cargar la web y esta encargado de inicializar a la misma.
 */
const init = () => {
  let navBar = document.getElementById("navBar");
   let footer = document.getElementById("footer");

  //Se carga el navbar
  navBar.innerHTML = Navbar();
  //Se carga footer
   footer.innerHTML = Footer();

  //Se le pasa al router la localización de la url en la que estamos.
  router(window.location.hash);

  //Cada vez que haya un cambio en la localización de la url, se le pasa de nuevo al router.
  window.addEventListener("hashchange", () => {
    router(window.location.hash);
  });
};

/**
 * Cuando la web carga, ejecuta el metodo init()
 */
window.addEventListener("load", init);

import { routes, routesName } from "./index";
import { setActive } from "./components/core/navbar/navbar";

/**
 * Este método funciona como router, se encargar de cargar los componentes según en que ruta estemos.
 * @param {*} route
 */
const router = async (route) => {
  // Busca detro del index.html donde se encuentra el elemento con id root
  let content = document.getElementById("root");
  //Lo vacía.
  content.innerHTML = "";

  setActive(route);

  // Un switch para que se rellene el elemento con id root con el componente designado a es ruta.
  switch (route) {
    case routesName.void:
      return content.appendChild(routes.home());

    case routesName.clear:
      return content.appendChild(routes.home());

    case routesName.home:
      return content.appendChild(routes.home());

    case routesName.dataset:
      return content.appendChild(routes.dataset());

    case routesName.sourceCode:
      let template;
      routes.sourceCode().then(function (v) {
        template = v.default();
        return content.appendChild(template);
      });
      break;

    case routesName.notFound:
      return content.appendChild(routes.notFound());

    default:
      //Si la ruta introducida no concuerda con ninguna de las anteriores, se redirige al 404 not found.
      var newUrl = window.location.protocol + `/${routesName.notFound}`;
      window.location.href = newUrl;
      return content.appendChild(routes.notFound());
  }
};

export { router };

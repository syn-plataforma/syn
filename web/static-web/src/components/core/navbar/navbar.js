import view from "./navbar.html";
import { routesName } from "../../../index";

/**
 * Función que carga el componente.
 */
export default () => {
  return view;
};

/**
 * Función que sirve para setear el elemento del navbar como activo según en que ruta se encuentre.
 * @param {*} route
 */
export function setActive(route) {
  let homeButton = document.getElementById("homeNav");
  let datasetsButton = document.getElementById("datasetsNav");
  let sourceCodeButton = document.getElementById("sourceCodeNav");
  switch (route) {
    case routesName.void:
      {
        if (!homeButton.classList.contains("active")) {
          homeButton.classList.add("active");
          datasetsButton.classList.remove("active");
          sourceCodeButton.classList.remove("active");
        }
      }
      break;

    case routesName.clear:
      {
        if (!homeButton.classList.contains("active")) {
          homeButton.classList.add("active");
          datasetsButton.classList.remove("active");
          sourceCodeButton.classList.remove("active");
        }
      }
      break;

    case routesName.home:
      {
        if (!homeButton.classList.contains("active")) {
          homeButton.classList.add("active");
          datasetsButton.classList.remove("active");
          sourceCodeButton.classList.remove("active");
        }
      }
      break;

    case routesName.dataset:
      {
        datasetsButton.classList.add("active");
        homeButton.classList.remove("active");
        sourceCodeButton.classList.remove("active");
      }
      break;

    case routesName.notFound:
      {
        homeButton.classList.remove("active");
        datasetsButton.classList.remove("active");
        sourceCodeButton.classList.remove("active");
      }
      break;

    case routesName.sourceCode:
      {
        sourceCodeButton.classList.add("active");
        homeButton.classList.remove("active");
        datasetsButton.classList.remove("active");
      }
      break;
  }
}

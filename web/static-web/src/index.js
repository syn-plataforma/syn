import Home from "./components/sections/home/home.controller";
import Dataset from "./components/sections/dataset/dataset.controller";
import NotFound from "./components/core/404/404.controller";

/**
 * Constant que contiene los controladores de cada componente.
 */
const routes = {
  home: Home,
  dataset: Dataset,
  notFound: NotFound,
  sourceCode: () =>
    Promise.resolve(
      import("./components/sections/sourceCode/sourceCode.controller")
    ),
};
/**
 * Constant para nombrar a cada ruta de la web.
 */
const routesName = {
  void: "",
  clear: "#/",
  home: "#/home",
  dataset: "#/datasets",
  notFound: "#/404",
  sourceCode: "#/sourceCode",
};

export { routes, routesName };

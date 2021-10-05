import view from "./sourceCode.html";

/**
 * Función que carga el componente.
 */
export default () => {
  //Crea un nuevo div dentro del root.
  const divElement = document.createElement("div");
  //Ese div lo rellena con el html del componente.
  divElement.innerHTML = view;

  //Se añaden las funcionalidades js.

  //Devuelve el componente renderizado.
  return divElement;
};

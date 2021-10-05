# Instrucciones

## Agregar componente a la web

### Creación de componente

Los componentes se deben de crear en './src/components/sections/nombrecomponente'
Se crean los archivos:

1. nombrecomponente.html
2. nombrecomponente.js
3. nombrecomponente.scss

El archivo scss debe importarse a './src/assets/sass/main.scss' para que los estilos del componente funcionen.

### Vinculación del componente con el router

Para vincular el nuevo componente al router, deberás ir a './src/index.js' y ahí deberas hacer 3 pasos:

1. Importar el componente.
2. Dar nombre al componente en la const routes
3. Crear la ruta del componente, asignadole a la misma el mismo nombre del componente.

Tras eso, deberás ir a './src/index.routes.js' y agregar al switch el nuevo componente dentro de un nuevo case.

### (Opcional) Agregar ruta al navbar

Para agregar la ruta al navbar simplemente, agregar un nuevo li con la ruta dada en el paso anterior a './src/components/core/navbar/navbar.html'

## Ejecución en modo desarrollo

```shell
npm start
```

## Construir el proyecto

```shell
npm run build
```

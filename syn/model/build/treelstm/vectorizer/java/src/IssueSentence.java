import org.bson.types.ObjectId;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Clase utilizada para definir los campos a recuperar y actualizar en las colecciones MongoDB que almacen los
 * conjuntos de datos.
 */
public class IssueSentence {

    // Nombres de las claves del HashMap<String, Double> que almacena los pesos de atención asociados a la hoja y el
    // contexto.
    public final static String EMBEDDING_LEAF_ELEMENT_NAME = "LEAF_WEIGHT";
    public final static String EMBEDDING_CONTEXT_ELEMENT_NAME = "CONTEXT_WEIGHT";

    // Etiquetas POS existentes en la rama que va desde el nodo hoja hasta el nodo raíz, excluyendo el nodo raíz.
    public ArrayList<List<String>> tokenBranch;

    // Árbol de constituyentes obtenido utilizando la librería de Stanford:
    //      edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser
    String constituentTree;

    // Etiquetas POS existentes en la rama que va desde el nodo hoja hasta el nodo raíz, excluyendo el nodo raíz y las
    // etiquetas no informativas.
    public ArrayList<List<String>> contextBranch;


    // Etiquetas de la rama del arbol binario generado a partir del árbol de constituyentes, para cada una de las
    // palabras existentes en el título o descripción de la incidencia. Para obtener las etiquetas se parte del nodo
    // hoja, o palabra, y se asciende en el árbol hasta llegar al nodo raiz.
    ArrayList<List<String>> constituentsTags;

    // Vectores numéricos de las etiquetas de la rama del arbol binario generado a partir del árbol de constituyentes,
    // para cada una de las palabras existentes en el título o descripción de la incidencia. Cada vector numérico es
    // un HashMap<String, Integer> creado utilizando la clase PartOfSpeech, en la que las claves son todas las
    // etiquetas generadas por el árbol binario de constituyentes, y los valores son el número de veces que aparece
    // cada clave en la rama asociada a cada palabra.
    ArrayList<HashMap<String, Double>> detailedConstituentsEmbeddings;

    // Vectores numéricos de las etiquetas de la rama del arbol binario generado a partir del árbol de constituyentes,
    // para cada una de las palabras existentes en el título o descripción de la incidencia. Cada vector numérico es
    // un ArrayList<Double> creado utilizando la clase PartOfSpeech, en la que el primer elemento
    // etiquetas generadas por el árbol binario de constituyentes, y los valores son el número de veces que aparece
    // cada clave en la rama asociada a cada palabra.
    ArrayList<ArrayList<Double>> constituentsEmbeddings;
}

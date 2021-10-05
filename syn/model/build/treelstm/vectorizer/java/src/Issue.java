import org.bson.types.ObjectId;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Clase utilizada para definir los campos NLP a recuperar y actualizar en las colecciones MongoDB que almacena los
 * conjuntos de datos.
 */
public class Issue {
    // Identificador único del documento en la BBDD.
    ObjectId _id;

    // Identificador de la incidencia.
    long bugId;

    // La incidencia ha sido rechazada por superar el número de tokens.
    boolean rejected;

    // Número de tokens existentes en el texto de la descripción de la incidencia.
    int totalNumTokens;

    // Número de tokens existentes en cada sentencia del texto de la descripción de la incidencia.
    ArrayList<Integer> tokens;

    // Tokens existentes en el texto de la descripción de la incidencia.
    ArrayList<ArrayList<String>> detailedTokens;

    // Lemmas existentes en el texto de la descripción de la incidencia.
    ArrayList<ArrayList<String>> detailedLemmas;

    // Número de sentencias existentes en el texto de la descripción de la incidencia.
    int numSentences;

    // Sentencias u oraciones obtenidas por la librería StanfordCoreNLP al procesar el texto existente en la descripción
    // de la incidencia con el "annotator" "ssplit".
    ArrayList<String> sentences;

    // Árboles sintácticos (constituency trees) obtenidos para cada una de las oraciones obtenidas al utilizar el
    // "annotator" "ssplit" de la librería StanfordCoreNLP en el texto existente en la descripción de la incidencia.
    ArrayList<String> constituencyTrees;

    // Árboles sintácticos binarios (constituency trees) obtenidos para cada una de las oraciones obtenidas al utilizar
    // el "annotator" "ssplit" de la librería StanfordCoreNLP en el texto existente en la descripción de la incidencia.
    ArrayList<String> binaryConstituencyTrees;

    // Árboles sintácticos (constituency trees) binarios colapsados obtenidos para cada una de las oraciones obtenidas
    // al utilizar el "annotator" "ssplit" de la librería StanfordCoreNLP en el texto existente en la descripción de la
    // incidencia. Se utilizan sólo las CoreLabels.
    ArrayList<String> collapsedBinaryConstituencyTrees;

    // Ramas de los árboles sintácticos (constituency trees) obtenidos para cada una de las oraciones obtenidas al
    // utilizar el "annotator" "ssplit" de la librería StanfordCoreNLP en el texto existente en la descripción de la
    // incidencia, y que conectan cada nodo hoja con el nodo raíz.
    ArrayList<ArrayList<List<String>>> tokenBranch;

    // Ramas de los árboles sintácticos (constituency trees) obtenidos para cada una de las oraciones obtenidas al
    // utilizar el "annotator" "ssplit" de la librería StanfordCoreNLP en el texto existente en la descripción de la
    // incidencia, y que conectan cada nodo hoja con el nodo raíz, eliminando el nodo raíz y los nodos no informativos.
    ArrayList<ArrayList<List<String>>> embeddingsContextBranch;

    // Vectores de atención generados para cada nodo hoja y para cada rama del contexto asociado a ese nodo. El primer
    // elemento del HashMap es el texto "LEAF_WEIGHT" o "CONTEXT_WEIGHT", y el segundo elemento es el valor de la
    // atención para ese elemento.
    ArrayList<ArrayList<HashMap<String, Double>>> detailedConstituentsEmbeddings;

    // Vectores de atención generados para cada nodo hoja y para cada rama del contexto asociado a ese nodo. El primer
    // elemento del ArrayList es el peso del nodo hoja, y el segundo elemento el peso del contexto.
    ArrayList<ArrayList<ArrayList<Double>>> constituentsEmbeddings;

    // ArrayList con la probabilidad de que la etiqueta POS generada para cada palabra de un texto sea correcta.
    ArrayList<Double> leafScore;

    // Número de etiquetas utilizado para generar los vectores de atención.
    int attentionPOSTags;

    // Número de etiquetas utilizado para generar los vectores de atención cuya probabilidad de haberse generado
    // de forma correcta sea superior al 70 %.
    int absoluteCoherence;

    // Porcentaje del número de etiquetas utilizado para generar los vectores de atención cuya probabilidad de haberse
    // generado de forma correcta sea superior al 70 %.
    double relativeCoherence;

    // Probabilidad logarítmica mínima para considerar que una etiqueta POS se ha generado correctamente.
    double coherenceThreshold = -0.35;
}

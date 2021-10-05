import java.util.ArrayList;
import java.util.List;

/**
 * Almacena las etiquetas gramaticales "parts-of-speech" del inglés, codificadas utilizando el standard de facto
 * Penn Treebank Project.
 */
public class ConstituentEmbedding {
    // POS tag del nodo hoja.
    public String leafTag = null;

    // Peso asignado al nodo Hoja.
    public double leafWeight = 0.0;

    // Probablidad de que la etiqueta POS generada sea correcta.
    public double leafScore = 0.0;

    // Nodos que conectan el nodo hoja con el nodo raíz.
    public ArrayList<String> contextBranch = new ArrayList<>();

    // Peso asignado a cada uno de los nodos que conecta el nodo hoja con el nodo raíz.
    public ArrayList<Double> contextWeight = new ArrayList<>();

    // Probabilidades de que las etiquetas POS generada para cada uno de los nodos que conecta el nodo hoja con el nodo
    // raiz, sea correcta.
    public ArrayList<Double> contextScore = new ArrayList<>();


    /**
     * Suma el peso de cada uno de los nodos que conecta el nodo hoja con el nodo raiz.
     *
     * @return double Suma de los pesos de la rama del nodo.
     */
    public double contextSum() {
        double sum = 0.0;
        for (double value : this.contextWeight) {
            sum += value;
        }
        return sum;
    }

    /**
     * Calcula la media de los pesos asociados a cada uno de los nodos que conecta el nodo hoja con el nodo raiz.
     *
     * @return double Media de los pesos de la rama del nodo.
     */
    public double contextAverage() {
        if (this.contextWeight.size() > 0) {
            return this.contextSum() / this.contextWeight.size();
        } else return 0.0;
    }

    /**
     * Para cada palabra calcula el peso de la palabra y el del contexto asociado.
     *
     * @return ArrayList<Double> El primer elemento es el peso de la hoja y el segundo elemento es el peso del contexto.
     */
    public ArrayList<Double> averageEmbedding() {
        ArrayList<Double> result = new ArrayList<>();
        result.add(this.leafWeight);
        result.add(this.contextAverage());
        return result;
    }

    /**
     * Suma las probabilidades de que las etiquetas POS generadas para cada nodo que conecta el nodo hoja con el nodo
     * raiz sea correcta.
     *
     * @return double Suma de las probabilidades de que las etiquetas POS de la rama del nodo sean correctas.
     */
    public double scoreSum() {
        double sum = 0.0;
        for (double value : this.contextScore) {
            sum += value;
        }
        return sum;
    }

    /**
     * Calcula la media de las probabilidades de que las etiquetas POS generadas para cada nodo que conecta el nodo hoja
     * con el nodo raiz sea correcta.
     *
     * @return double Mediade las probabilidades de que las etiquetas POS de la rama del nodo sean correctas.
     */
    public double scoreAverage() {
        if (this.contextScore.size() > 0) {
            return this.scoreSum() / this.contextScore.size();
        } else return 0.0;
    }

    /**
     * Para cada palabra obtiene la probabilidad de que la etiqueta POS generada sea correcta y obtiene la media de las
     * probabilidades de que las etiquetas POS generadas para cada nodo que conecta el nodo hoja con el nodo raiz sean
     * correctas.
     *
     * @return ArrayList<Double> El primer elemento es la probabilidad de que la  etiqueta POS generada sea correcta y
     * el segundo elemento es la media de las probabilidades de que las etiquetas POS generadas para el contexto sean
     * correctas.
     */
    public ArrayList<Double> averageScore() {
        ArrayList<Double> result = new ArrayList<>();
        result.add(this.leafScore);
        result.add(this.scoreAverage());
        return result;
    }
}
import java.util.HashMap;

/**
 * Almacena las etiquetas gramaticales "parts-of-speech" del inglés, codificadas utilizando el standard de facto
 * Penn Treebank Project.
 */
public enum WordWeight {

    // Peso máximo.
    MAX_WEIGHT("MaxWeight", 1.0),

    //
    //   Peso de cada clase de palabra.
    //

    CORE_WORD("CoreWord", 1.0 * MAX_WEIGHT.weight),
    MODIFIER_WORD("ModifierWord", 0.75 * MAX_WEIGHT.weight),
    FUNCTION_WORD("FunctionWord", 0.5 * MAX_WEIGHT.weight),
    NON_WORD("NonWord", 0.25 * MAX_WEIGHT.weight);

    // Atributo peso.
    private final String name;
    private final double weight;

    /**
     * Constructor
     *
     * @param weight double Peso de la clase de palabra.
     */
    private WordWeight(String name, double weight) {
        this.name = name;
        this.weight = weight;
    }

    /**
     * Devuelve el peso asiganado a la clase de palabra.
     *
     * @return double Peso de la clase de palabra.
     */
    public String getName() {
        return name;
    }

    public double getWeight() {
        return weight;
    }

    public static double getWeight(String value) {
        for (WordWeight v : values()) {
            if (value.equals(v.getName())) {
                return v.getWeight();
            }
        }

        throw new IllegalArgumentException("Unknown word class: '" + value + "'.");
    }
}
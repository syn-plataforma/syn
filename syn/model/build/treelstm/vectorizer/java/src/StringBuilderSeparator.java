/**
 * Almacena las etiquetas gramaticales "parts-of-speech" del inglés, codificadas utilizando el standard de facto
 * Penn Treebank Project.
 */
public enum StringBuilderSeparator {
    //
    //   Separadores utilizados en el StringBuilder.
    //

    ATTRIBUTE(" ATTRIBUTE "),
    SENTENCE(" SENTENCE "),
    ELEMENT("\t");


    private final String value;

    private StringBuilderSeparator(String value) {
        this.value = value;
    }

    /**
     * Devuelve el valor del separador.
     *
     * @return String Separador.
     */
    public String toString() {
        return getValue();
    }

    /**
     * Devuelve el valor del separador.
     *
     * @return String Separador.
     */
    protected String getValue() {
        return this.value;
    }

    public static StringBuilderSeparator get(String value) {
        for (StringBuilderSeparator v : values()) {
            if (value.equals(v.getValue())) {
                return v;
            }
        }

        throw new IllegalArgumentException("Nombre separador desconocido: '" + value + "'.");
    }
}
import edu.stanford.nlp.trees.Tree;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Almacena las etiquetas gramaticales "parts-of-speech" del inglés, codificadas utilizando el standard de facto
 * Penn Treebank Project.
 */
public enum PartOfSpeech {
    //
    //   Clause Level
    //

    SIMPLE_DECLARATIVE_CLAUSE("S", "NonInformativeWord"),
    CLAUSE_INTRODUCED_BY_A_SUBORDINATING_CONJUNCTION("SBAR", "NonInformativeWord"),
    DIRECT_QUESTION_INTRODUCED_BY_A_WH_WORD_OR_A_WH_PHRASE("SBARQ", "NonInformativeWord"),
    INVERTED_DECLARATIVE_SENTENCE("SINV", "NonInformativeWord"),
    INVERTED_YES_NO_QUESTION("SQ", "NonInformativeWord"),

    //
    //   Phrase Level
    //

    ADJECTIVE_PHRASE("ADJP", "ModifierWord"),
    ADVERB_PHRASE("ADVP", "ModifierWord"),
    CONJUNCTION_PHRASE("CONJP", "FunctionWord"),
    FRAGMENT("FRAG", "CoreWord"),
    INTERJECTION("INTJ", "ModifierWord"),
    LIST_MARKER("LST", "CoreWord"),
    NOT_A_CONSTITUENT("NAC", "CoreWord"),
    NOUN_PHRASE("NP", "CoreWord"),
    HEAD_OF_THE_NP("NX", "CoreWord"),
    PREPOSITIONAL_PHRASE("PP", "FunctionWord"),
    PARENTHETICAL("PRN", "CoreWord"),
    PARTICLE("PRT", "FunctionWord"),
    QUANTIFIER_PHRASE("QP", "CoreWord"),
    REDUCED_RELATIVE_CLAUSE("RRC", "CoreWord"),
    UNLIKE_COORDINATED_PHRASE("UCP", "CoreWord"),
    VERB_PHRASE("VP", "CoreWord"),
    WH_ADJECTIVE_PHRASE("WHADJP", "ModifierWord"),
    WH_ADVERB_PHRASE("WHADVP", "ModifierWord"),
    WH_NOUN_PHRASE("WHNP", "CoreWord"),
    WH_PREPOSITIONAL_PHRASE("WHPP", "FunctionWord"),
    UNKNOWN("X", "CoreWord"),

    //
    //   Word level
    //

    COORDINATING_CONJUNCTION("CC", "FunctionWord"),
    CARDINAL_NUMBER("CD", "NonWord"),
    DETERMINER("DT", "FunctionWord"),
    EXISTENTIAL_THERE("EX", "FunctionWord"),
    FOREIGN_WORD("FW", "CoreWord"),
    PREPOSITION_OR_SUBORDINATING_CONJUNCTION("IN", "FunctionWord"),
    ADJECTIVE("JJ", "ModifierWord"),
    ADJECTIVE_COMPARATIVE("JJR", "ModifierWord"),
    ADJECTIVE_SUPERLATIVE("JJS", "ModifierWord"),
    LIST_ITEM_MARKER("LS", "CoreWord"),
    MODAL("MD", "ModifierWord"),
    NOUN_SINGULAR_OR_MASS("NN", "CoreWord"),
    NOUN_PLURAL("NNS", "CoreWord"),
    PROPER_NOUN_SINGULAR("NNP", "CoreWord"),
    PROPER_NOUN_PLURAL("NNPS", "CoreWord"),
    PREDETERMINER("PDT", "FunctionWord"),
    POSSESSIVE_ENDING("POS", "CoreWord"),
    PERSONAL_PRONOUN("PRP", "FunctionWord"),
    POSSESSIVE_PRONOUN("PRP$", "FunctionWord"),
    ADVERB("RB", "ModifierWord"),
    ADVERB_COMPARATIVE("RBR", "ModifierWord"),
    ADVERB_SUPERLATIVE("RBS", "ModifierWord"),
    WORD_PARTICLE("RP", "FunctionWord"),
    SYMBOL("SYM", "NonWord"),
    TO("TO", "FunctionWord"),
    WORD_INTERJECTION("UH", "ModifierWord"),
    VERB_BASE_FORM("VB", "CoreWord"),
    VERB_PAST_TENSE("VBD", "CoreWord"),
    VERB_GERUND_OR_PRESENT("VBG", "CoreWord"),
    VERB_PAST_PARTICIPLE("VBN", "CoreWord"),
    VERB_NON_3RD_PERSON_SINGULAR_PRESENT("VBP", "CoreWord"),
    VERB_3RD_PERSON_SINGULAR_PRESENT("VBZ", "CoreWord"),
    WH_DETERMINER("WDT", "CoreWord"),
    WH_PRONOUN("WP", "CoreWord"),
    POSSESSIVE_WH_PRONOUN("WP$", "CoreWord"),
    WH_ADVERB("WRB", "CoreWord"),

    //
    //   Function tags
    //

    // Form/function discrepancies

    ADVERBIAL("-ADV", "NonInformativeWord"),
    NOMINAL("-NOM", "NonInformativeWord"),

    // Grammatical role

    DATIVE("-DTV", "NonInformativeWord"),
    LOGICAL_SUBJECT("-LGS", "NonInformativeWord"),
    PREDICATE("-PRD", "NonInformativeWord"),
    LOCATIVE_COMPLEMENT_OF_PUT("PUT", "NonInformativeWord"),
    SURFACE_SUBJECT("-SBJ", "NonInformativeWord"),
    TOPICALIZED("-TPC", "NonInformativeWord"),
    VOCATIVE("-VOC", "NonInformativeWord"),

    // Adverbials

    BENEFACTIVE("-BNF", "NonInformativeWord"),
    DIRECTION("-DIR", "NonInformativeWord"),
    EXTENT("-EXT", "NonInformativeWord"),
    LOCATIVE("-LOC", "NonInformativeWord"),
    MANNER("-MNR", "NonInformativeWord"),
    PURPOSE_OR_REASON("-PRP", "NonInformativeWord"),
    TEMPORAL("-TMP", "NonInformativeWord"),

    // Miscellaneous

    CLOSELY_RELATED("-CLR", "NonInformativeWord"),
    CLEFT("-CLF", "NonInformativeWord"),
    HEADLINE("-HLN", "NonInformativeWord"),
    TITLE("-TTL", "NonInformativeWord"),

    //
    //  TreeBinarizer
    //

    TREE_BINARIZER_GW("GW", "NonInformativeWord"),
    TREE_BINARIZER_ADD("ADD", "NonInformativeWord"),
    TREE_BINARIZER_NFP("NFP", "NonInformativeWord"),
    TREE_BINARIZER_AFX("AFX", "NonInformativeWord"),
    TREE_BINARIZER_HYPH("HYPH", "NonInformativeWord"),

    //
    //   Stanford
    //

    //    SENTENCE_SEPARATOR(",", "NonInformativeWord"),
    SENTENCE_TERMINATOR(".", "NonWord");


    // Etiqueta POS.
    private final String tag;

    // Tipo de la etiqueta POS definido en el mecanismo de atención.
    private final String wordClass;

    /**
     * Constructor de la clase.
     *
     * @param tag       String Etiqueta POS.
     * @param wordClass String Tipo de la etiqueta POS.
     */
    PartOfSpeech(String tag, String wordClass) {
        this.tag = tag;
        this.wordClass = wordClass;
    }


    /**
     * Devuelve la codificación correspondiente a la etiqueta gramtical o part-of-speech..
     *
     * @return String código Penn Treebank para el inglés de la etiqueta gramatical o part-of-speech.
     */
    public String toString() {
        return getTag();
    }

    /**
     * Devuelve la etiqueta POS.
     *
     * @return String Etiqueta POS.
     */
    protected String getTag() {
        return this.tag;
    }

    /**
     * Devuelve el tipo de la etiqueta POS.
     *
     * @return String Tipo de la etiqueta POS.
     */
    private String wordClass() {
        return wordClass;
    }

    /**
     * Comprueba si la etiqueta POS existe en el conjunto de etiquetas definidas.
     *
     * @param value String Etiqueta POS.
     * @return PartOfSpeech Etiqueta POS asociada al nombre de la etiqueta POS.
     */
    public static PartOfSpeech get(String value) {
        for (PartOfSpeech v : values()) {
            if (value.equals(v.getTag())) {
                return v;
            }
        }

        throw new IllegalArgumentException("Unknown part of speech: '" + value + "'.");
    }

    /**
     * Inicializa un HashMap cuyas claves con los códigos Penn Treebank para el inglés de todas las etiquetas o
     * part-of-speech, y cuyos valores almacenarán el número de veces que aparece la etitqueta en una rama.
     *
     * @return HashMap<String, Integer> cuyas claves son los códigos Penn Treebank para el inglés de la etiqueta
     * gramatical o part-of-speech, y cuyos valores son todos ceros.
     */
    public static HashMap<String, Integer> initializeEmbedding() {
        HashMap<String, Integer> embedding = new HashMap<>();
        for (PartOfSpeech v : values()) {
            embedding.put(v.getTag(), 0);
        }
        return embedding;
    }

    /**
     * Objeto HashMap cuyas claves con los códigos Penn Treebank para el inglés de todas las etiquetas o
     * part-of-speech, y cuyos valores almacenarán el número de veces que aparece la etitqueta en una rama.
     *
     * @return HashMap<String, Integer> cuyas claves son los códigos Penn Treebank para el inglés de la etiqueta
     * gramatical o part-of-speech, y cuyos valores con el número de veces que aparece la etitqueta en una rama.
     */
    public static HashMap<String, Integer> getEmbedding(HashMap<String, Integer> embedding, String value) {
        for (PartOfSpeech v : values()) {
            if (value.equals(v.getTag()) || value.equals("@".concat(v.getTag()))) {
                embedding.put(v.getTag(), embedding.get(v.getTag()) + 1);
                return embedding;
            }
        }

        embedding.put(UNKNOWN.getTag(), embedding.get(UNKNOWN.getTag()) + 1);
        return embedding;
    }


    /**
     * Actualiza el valor del peso de la atención asociado a un nodo hoja.
     *
     * @param embedding ConstituentEmbedding Objeto en el que se actualizará el peso de la atención  asociado al nodo
     *                  hoja.
     * @param value     String Etiqueta POS asociada al nodo hoja.
     * @return ConstituentEmbedding Objeto en el que se ha actualizado el peso de la atención asociado al nodo hoja.
     */
    public static ConstituentEmbedding updateLeafWeight(ConstituentEmbedding embedding, String value) {
        for (PartOfSpeech v : values()) {
            if (value.equals(v.getTag()) && !"NonInformativeWord".equals(v.wordClass)) {
                embedding.leafTag = v.getTag();
                embedding.leafWeight = WordWeight.getWeight(v.wordClass);
                return embedding;
            }
        }

        return embedding;
    }

    /**
     * Actualiza el valor del peso de la atención asociado al contexto de un nodo hoja.
     *
     * @param embedding ConstituentEmbedding Objeto en el que se actualizará el peso de la atención  asociado al
     *                  contexto de un nodo hoja.
     * @param value     String Etiqueta POS asociada al nodo perteneciente al contexto del nodo hoja.
     * @return ConstituentEmbedding Objeto actualizado con el peso de la atención  asociado al contexto de una hoja.
     */
    public static ConstituentEmbedding updateContextWeight(ConstituentEmbedding embedding, String value) {
        for (PartOfSpeech v : values()) {
            if (value.equals(v.getTag()) && !"NonInformativeWord".equals(v.wordClass)) {
                embedding.contextBranch.add(v.tag);
                embedding.contextWeight.add(WordWeight.getWeight(v.wordClass));
                return embedding;
            }
        }

        return embedding;
    }

    /**
     * Obtiene la probabilidad de que la etiqueta POS asociada a un nodo sea correcta.
     *
     * @param tree  Tree Nodo para el que se va a calcular la probabilidad de que la etiqueta POS que le ha asoicado el
     *              parser utilizado sea correcta.
     * @param value String Etiqueta POS asociada al nodo hoja.
     * @return Double Probabilidad de que la etiqueta POS asociada a un nodo sea correcta.
     */
    public static Double getLeafScore(Tree tree, String value) {
        for (PartOfSpeech v : values()) {
            if (value.equals(v.getTag()) && !"NonInformativeWord".equals(v.wordClass)) {
                return tree.score();
            }
        }

        return Double.NaN;
    }
}
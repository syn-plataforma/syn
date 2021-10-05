import com.mongodb.MongoException;
import com.mongodb.bulk.BulkWriteResult;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.BulkWriteOptions;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.UpdateOneModel;
import com.mongodb.client.model.WriteModel;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import org.bson.Document;
import org.bson.conversions.Bson;
import org.bson.types.ObjectId;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * Recupera el título o la descripción de una incidencia existente en una colección MongoDB y crea el árbol binario de
 * constituyentes, las etiquetas gramaticales o part-of-speech, correspondientes a cada una de las palabras existentes
 * en el texto, los vectores numéricos o tag embeddings, correspondientes a cada una de las palabras existentes en el
 * texto, y almacena esta información en un fichero de texto.
 */
public class UpdateMongoDBNLPFields {
    private final String host;
    private final int port;
    private final String dbName;
    private final String collName;
    private final String startYear;
    private final String endYear;
    private final String textColumnName;
    private final int maxNumTokens;
    private final String parserModel;
    private final boolean createTrees;
    private final boolean calcEmbeddings;
    private final boolean calcCoherence;
    private final int numThreads;

    private final CollapseUnaryTransformer transformer;

//    private static final String PCFG_PATH = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";


    /**
     * Constructor para inicializar los valores de los atributos con los parámetros de entrada.
     * <p>
     */
    public UpdateMongoDBNLPFields(
            String host,
            int port,
            String dbName,
            String collName,
            String startYear,
            String endYear,
            String textColumnName,
            int maxNumTokens,
            String parserModel,
            boolean createTrees,
            boolean calcEmbeddings,
            boolean calcCoherence,
            int numThreads
    ) {
        this.host = host;
        this.port = port;
        this.dbName = dbName;
        this.collName = collName;
        this.startYear = startYear;
        this.endYear = endYear;
        this.textColumnName = textColumnName;
        this.maxNumTokens = maxNumTokens;
        this.parserModel = parserModel;
        this.createTrees = createTrees;
        this.calcEmbeddings = calcEmbeddings;
        this.calcCoherence = calcCoherence;
        this.numThreads = numThreads;

        this.transformer = new CollapseUnaryTransformer();
    }


    public Tree collapseBinaryTree(Tree tree) {
        Tree collapsedUnary = transformer.transformTree(tree);
        // Converts the tree labels to CoreLabels.
        Trees.convertToCoreLabels(collapsedUnary);
        collapsedUnary.indexSpans();

        return collapsedUnary;
    }

    public IssueSentence issueSentence(IssueSentence sentence, Tree tree) {
        // Inicializa los objetos para almacenar las etiquetas de los costituyentes.
        sentence.tokenBranch = new ArrayList<>();
        sentence.contextBranch = new ArrayList<>();
        List<String> wordTags = new ArrayList<>();

        // Inicializa los objetos para almacenar los embeddings de cada palabra de las etiquetas de los constituyentes.
        ArrayList<ConstituentEmbedding> sentenceDetailedConstituentsEmbeddings = new ArrayList<>();

        // Obtiene las hojas para recorrer el árbol desde las hojas hasta la raíz.
        List<Tree> leaves = tree.getLeaves();

        // Recorre el árbol a partir de las hojas para obtener las etiquetas y los embeddings.
        for (Tree leaf : leaves) {
            wordTags.clear();
            ConstituentEmbedding wordEmbedding = new ConstituentEmbedding();

            // Obtiene el nodo preterminal.
            Tree cur = leaf.parent(tree);
            // Si  existe un nodo padre, se añade la etiqueta y se genera el embedding.
            if (cur != null && cur.label() != null) {
                // La etiqueta ROOT está presente en todas las ramas en el nodo raíz y por tanto no genera información
                // diferenciadora para las distitntas ramas, y por eso no se añade.
                if (!"ROOT".equals(cur.label().value())) {
                    wordTags.add(cur.label().toString());
                    PartOfSpeech.updateLeafWeight(wordEmbedding, cur.label().toString().replace("-", "").replaceAll("[0-9]", ""));
                }

                // Recorre el árbol ascendiendo hasta que llega al nodo raíz, que es cuando no se encuentra un padre.
                while (true) {
                    // Actualiza el nodo padre con el nodo actual.
                    Tree parent = cur.parent(tree);
                    // Si no existe padre se termina el bucle.
                    if (parent == null) {
                        break;
                    }

                    // Si existe el padre y no es el nodo raíz se añade la etiqueta y se calcula el embedding.
                    if (!"ROOT".equals(parent.label().value())) {
                        wordTags.add(parent.label().toString());
                        PartOfSpeech.updateContextWeight(wordEmbedding, parent.label().toString().replace("-", "").replaceAll("[0-9]", ""));
                    }

                    // Actualiza el nodo actual con el nodo padre.
                    cur = parent;
                }
                // Añade las etiquetas y los embeddings al resultado.
                sentence.tokenBranch.add(new ArrayList<>(wordTags));
//                wordEmbedding.contextBranch.add(new ArrayList<>(wordTags));
                sentenceDetailedConstituentsEmbeddings.add(wordEmbedding);
            }
        }

        // Comprueba si los Arrays están vacíos, ya que pueden llegar árboles vacíos.
        if (!sentenceDetailedConstituentsEmbeddings.isEmpty()) {
            sentence.detailedConstituentsEmbeddings = new ArrayList<>();
            HashMap<String, Double> detailedLocalEmbedding = new HashMap<>();

            sentence.constituentsEmbeddings = new ArrayList<>();
            for (ConstituentEmbedding elem : sentenceDetailedConstituentsEmbeddings) {
                detailedLocalEmbedding.clear();
                detailedLocalEmbedding.put(IssueSentence.EMBEDDING_LEAF_ELEMENT_NAME, elem.leafWeight);
                detailedLocalEmbedding.put(IssueSentence.EMBEDDING_CONTEXT_ELEMENT_NAME, elem.contextAverage());
                sentence.detailedConstituentsEmbeddings.add(new HashMap<>(detailedLocalEmbedding));

                sentence.constituentsEmbeddings.add(new ArrayList<>(elem.averageEmbedding()));

                sentence.contextBranch.add(new ArrayList<>(elem.contextBranch));

            }
            sentence.tokenBranch.add(new ArrayList<>(wordTags));
        }

        return sentence;
    }

    public void calculateCoherence(Tree tree, Issue issue) {
        // Obtiene las hojas para recorrer el árbol desde las hojas hasta la raíz.
        List<Tree> leaves = tree.getLeaves();

        for (Tree leaf : leaves) {
            Tree cur = leaf.parent(tree);
            if (!"ROOT".equals(leaf.label().value())) {
                Double score = PartOfSpeech.getLeafScore(cur, cur.label().toString().replace("-", "").replaceAll("[0-9]", ""));
                if (!score.isNaN()) {
                    issue.attentionPOSTags++;
                    issue.leafScore.add(score);
                    if (score > issue.coherenceThreshold) {
                        issue.absoluteCoherence++;
                    }
                }
            }
        }

        if (issue.attentionPOSTags > 0.0) {
            issue.relativeCoherence = (double) issue.absoluteCoherence / (double) issue.attentionPOSTags;
        }
    }

    /**
     * Procesa los documentos recuperados de MongDB.
     */
    public static void processDocuments(
            MongoCollection<Document> collection,
            Bson query,
            UpdateMongoDBNLPFields processor,
            StanfordCoreNLP tokensPipeline,
            StanfordCoreNLP sentencesPipeline,
            StanfordCoreNLP pipeline
    ) {
        long queryCount = collection.countDocuments(query);
        System.out.println("Documentos recuperados: " + queryCount);

        // Contador para el control del número de documentos leídos y procesados.
        AtomicInteger count = new AtomicInteger();

        AtomicInteger updateCount = new AtomicInteger();
        AtomicInteger updateRejectedCount = new AtomicInteger();

        long start = System.currentTimeMillis();

        // Capacidad inicial del buffer de escritura y procesamiento de incidencias.
        double initialCapacity = 1000.0;

        // Almacena las incidencias para hacer una actualización en modo bulk.
        List<WriteModel<Document>> updateList = new ArrayList<>((int) initialCapacity);

        try (MongoCursor<Document> cursor = collection.find(query)
                .projection(new Document("_id", 1)
                        .append("bug_id", 1)
                        .append("rejected", 1)
                        .append("total_num_tokens", 1)
                        .append("tokens", 1)
                        .append(processor.textColumnName, 1))
                .noCursorTimeout(true).cursor()) {
            while (cursor.hasNext()) {
                Document x = cursor.next();
                count.getAndIncrement();
                System.out.println("Iteración " + count.get() + " - bug_id: " + x.get("bug_id").toString());
                if (
                        (x.containsKey("rejected") && !x.getBoolean("rejected")) ||
                                (
                                        !x.containsKey("rejected") &&
                                                x.get(processor.textColumnName) instanceof String &&
                                                !x.getString(processor.textColumnName).trim().equals("")
                                )
                ) {
                    // Lee el texto.
                    String text = x.getString(processor.textColumnName);

                    // Inicializa el objeto Issue.
                    Issue issue = new Issue();

                    // Asigna el identificador.
                    issue._id = x.getObjectId("_id");
                    issue.bugId = Long.parseLong(x.get("bug_id").toString());
                    issue.rejected = false;
                    issue.totalNumTokens = 0;
                    issue.tokens = new ArrayList<>();
                    issue.detailedTokens = new ArrayList<>();
                    issue.detailedLemmas = new ArrayList<>();
                    issue.sentences = new ArrayList<>();
                    issue.constituencyTrees = new ArrayList<>();
                    issue.binaryConstituencyTrees = new ArrayList<>();
                    issue.collapsedBinaryConstituencyTrees = new ArrayList<>();
                    issue.tokenBranch = new ArrayList<>();
                    issue.embeddingsContextBranch = new ArrayList<>();
                    issue.detailedConstituentsEmbeddings = new ArrayList<>();
                    issue.constituentsEmbeddings = new ArrayList<>();
                    issue.leafScore = new ArrayList<>();

                    // Ejecuta todos los anotadores en el texto.
                    Annotation document = new Annotation(text);

                    tokensPipeline.annotate(document);

                    // these are all the sentences in this document- A CoreMap is essentially a Map that
                    // uses class objects as keys and has values with custom types
                    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

                    issue.numSentences = sentences.size();

                    // Lee el número de tokens si existe el campo tokens.
                    int sentenceNumTokens = 0;
                    if (!x.containsKey("tokens")) {
                        for (CoreMap sent : sentences) {
                            // Define el objeto sobre el que se van a ejecutar los anotadores.
                            Annotation sentAnnotation = new Annotation(sent.toString());

                            // Ejecuta todos los anotadores en cada sentencia.
                            sentencesPipeline.annotate(sentAnnotation);
                            issue.tokens.add(sentAnnotation.get(CoreAnnotations.TokensAnnotation.class).size());
                            if (sentAnnotation.get(CoreAnnotations.TokensAnnotation.class).size() > processor.maxNumTokens) {
                                sentenceNumTokens = sentAnnotation.get(CoreAnnotations.TokensAnnotation.class).size();
                                updateRejectedCount.getAndIncrement();
                                issue.rejected = true;
                            }
                            issue.totalNumTokens += sentAnnotation.get(CoreAnnotations.TokensAnnotation.class).size();
                        }
                    } else {
                        if (null != x.get("tokens") && "" != x.get("tokens")) {
                            issue.tokens = new ArrayList<>(x.getList("tokens", Integer.class));
                            if (x.containsKey("total_num_tokens") && x.getInteger("total_num_tokens") > 0) {
                                issue.totalNumTokens = x.getInteger("total_num_tokens");
                            } else {
                                for (Integer numtokens : issue.tokens) {
                                    issue.totalNumTokens += numtokens;
                                }
                            }
                            if (x.containsKey("rejected")) {
                                issue.rejected = x.getBoolean("rejected");
                            }
                        }
                    }

                    // Para algunas descripciones con un número elevado de tokens, el tiempo para generar
                    // los árboles es demasiado elevado y por tanto se utilizarán sólo aquellas incidencas
                    // cuyo número de tokens sea menor que 150. Se introducirá por tanto este filtro
                    // para los siguientes pasos, con el número de tokens como un argumento de entrada.
                    if (!issue.rejected && processor.createTrees) {
                        // NLP
                        for (CoreMap sentence : sentences) {
                            // Define el objeto sobre el que se van a ejecutar los anotadores.
                            Annotation sentenceAnnotation = new Annotation(sentence.toString());

                            // Ejecuta todos los anotadores en cada sentencia.
                            pipeline.annotate(sentenceAnnotation);

                            List<CoreLabel> sentenceTokens = sentenceAnnotation.get(CoreAnnotations.TokensAnnotation.class);

                            // Añade los tokes y lemmas de cada sentencia al objeto issue.
                            ArrayList<String> sentenceTokensAsString = new ArrayList<>();
                            ArrayList<String> sentenceLemmasAsString = new ArrayList<>();
                            for (CoreLabel token : sentenceTokens) {
                                sentenceTokensAsString.add(token.value());
                                sentenceLemmasAsString.add(token.lemma());
                            }
                            issue.detailedTokens.add(sentenceTokensAsString);
                            issue.detailedLemmas.add(sentenceLemmasAsString);

                            // Añade la sentencia al objeto issue.
                            List<CoreMap> sentencesFromSentenceAnnotation = sentenceAnnotation.get(CoreAnnotations.SentencesAnnotation.class);
                            CoreMap sentenceFromSentenceAnnotation = sentencesFromSentenceAnnotation.get(0);
                            issue.sentences.add(sentenceFromSentenceAnnotation.toString());

                            // Obtiene los árboles.
                            Tree tree = sentenceFromSentenceAnnotation.get(TreeCoreAnnotations.TreeAnnotation.class);
                            Tree binaryTree = sentenceFromSentenceAnnotation.get(TreeCoreAnnotations.BinarizedTreeAnnotation.class);
                            Tree collapsedUnaryTree = processor.collapseBinaryTree(binaryTree);

                            issue.constituencyTrees.add(tree.toString());
                            issue.binaryConstituencyTrees.add(binaryTree.toString());
                            issue.collapsedBinaryConstituencyTrees.add(collapsedUnaryTree.toString());

                            // Calcula los embeddings.
                            if (processor.calcEmbeddings) {
                                IssueSentence issueSentence = processor.issueSentence(new IssueSentence(), collapsedUnaryTree);

                                issue.tokenBranch.add(issueSentence.tokenBranch);
                                issue.embeddingsContextBranch.add(issueSentence.contextBranch);
                                issue.detailedConstituentsEmbeddings.add(issueSentence.detailedConstituentsEmbeddings);
                                issue.constituentsEmbeddings.add(issueSentence.constituentsEmbeddings);
                            }

                            // Calcula la coherencia.
                            if (processor.calcCoherence) {
                                processor.calculateCoherence(tree, issue);
                            }
                        }
                    } else {
                        if (issue.rejected) {
                            System.err.printf("Documentos rechazados =  %d \n", updateRejectedCount.get());
                            System.err.println("Iteración " + count.get() + " - bug_id rechazado: " + x.get("bug_id").toString() + " - sentence tokens: " + sentenceNumTokens);
                        }
                    }

                    // Acutualiza la colección Mongodb.
//                                System.err.println("Iteración " + updateCount.get() + " - bug_id: " + issue.bugId);

                    // Parte del documento que se actualiza siempre.
                    Document sentenceDocument = new Document("sentences", issue.sentences)
                            .append("num_sentences", issue.numSentences)
                            .append("rejected", issue.rejected)
                            .append("total_num_tokens", issue.totalNumTokens)
                            .append("tokens", issue.tokens);

                    // Parte del documento que se actualiza si createTrees es true.
                    if (!issue.rejected && processor.createTrees) {
                        sentenceDocument.append("detailed_tokens", issue.detailedTokens)
                                .append("detailed_lemmas", issue.detailedLemmas)
                                .append("constituency_trees", issue.constituencyTrees)
                                .append("binary_constituency_trees", issue.binaryConstituencyTrees)
                                .append("collapsed_binary_constituency_trees", issue.collapsedBinaryConstituencyTrees);
                    }

                    // Parte del documento que se actualiza si calcEmbeddings es true.
                    if (processor.calcEmbeddings) {
                        sentenceDocument.append("token_branch", issue.tokenBranch)
                                .append("embeddings_context_branch", issue.embeddingsContextBranch)
                                .append("detailed_constituent_embeddings", issue.detailedConstituentsEmbeddings)
                                .append("constituents_embeddings", issue.constituentsEmbeddings);
                    }

                    // Parte del documento que se actualiza si calcCoherence es true.
                    if (processor.calcCoherence) {
                        sentenceDocument.append("leaf_score", issue.leafScore)
                                .append("attention_pos_tags", issue.attentionPOSTags)
                                .append("absolute_coherence", issue.absoluteCoherence)
                                .append("relative_coherence", issue.relativeCoherence);
                    }

                    updateList.add(
                            new UpdateOneModel<>(
                                    new Document("_id", new ObjectId(issue._id.toString())), // filter
                                    new Document("$set", sentenceDocument
                                    )
                            )
                    );


                    if (issue.rejected) {
                        updateRejectedCount.getAndIncrement();
                        System.err.printf("Documentos rechazados =  %d \n", updateRejectedCount.get());
                        System.err.println("Iteración " + count.get() + " - bug_id rechazado: " + x.get("bug_id").toString() + " - sentence tokens: " + sentenceNumTokens);
                    }

                    if ((updateList.size() > 0) && (((updateList.size() % initialCapacity) == 0) || (count.get() == queryCount))) {
                        BulkWriteResult result = collection.bulkWrite(updateList, new BulkWriteOptions().ordered(false));
                        System.err.printf("Documentos actualizados en MongoDB: %d \n", result.getMatchedCount());
                        updateList.clear();

                        double elapsed = (System.currentTimeMillis() - start) / initialCapacity;
                        System.err.printf("Documentos procesados: %d (%.2f seg)\n", updateCount.get(), elapsed);
                    }
                } else {
                    updateRejectedCount.getAndIncrement();
                    System.err.println("Iteración: " + count.get() + " - bug_id rechazado: " + x.get("bug_id").toString());
                }
            }
        }

        long totalTimeMillis = System.currentTimeMillis() - start;
        System.err.printf("Actualizados %d documentos en %.2f seg (%.1fm seg por documento)\n",
                updateCount.get(), totalTimeMillis / initialCapacity, totalTimeMillis / (double) count.get());
        System.err.println("Incidencias rechazadas: " + updateRejectedCount.get());
    }

    /**
     * Cierra todos Writers abiertos.
     */
    public static void main(String[] args) {
        // Cambia la codificación de System.out a UTF-8.
        try {
            System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true, "UTF-8"));
            System.setErr(new PrintStream(new FileOutputStream(FileDescriptor.err), true, "UTF-8"));
        } catch (UnsupportedEncodingException e) {
            throw new InternalError("Codificación UTF-8 no soportada.");
        }

        Properties props = StringUtils.argsToProperties(args);

        // Argumentos de entrada.
        UpdateMongoDBNLPFields processor = new UpdateMongoDBNLPFields(
                props.getProperty("host", "localhost"),
                Integer.parseInt(props.getProperty("port", "30017")),
                props.getProperty("dbName", "eclipse"),
                props.getProperty("collName", "normalized_clear"),
                props.getProperty("startYear", "2000"),
                props.getProperty("endYear", "2021"),
                props.getProperty("textColumnName", "normalized_description"),
                Integer.parseInt(props.getProperty("maxNumTokens", "150")),
                props.getProperty("parserModel", "corenlp"),
                Boolean.parseBoolean(props.getProperty("createTrees", Boolean.toString(true))),
                Boolean.parseBoolean(props.getProperty("calcEmbeddings", Boolean.toString(true))),
                Boolean.parseBoolean(props.getProperty("calcCoherence", Boolean.toString(true))),
                Integer.parseInt(props.getProperty("numThreads", "8"))
        );

        // Para no tener un error de tipo OutOfMemoryError, es necesario cnoocer primero el número de tokens, y por
        // tanto se crea una primera pipeline para obtener los tokens y después filtrar por el número de tokens.
        Properties tokensProperties = new Properties();
        tokensProperties.put("annotators", "tokenize, ssplit");
        // If non-null value is a String which contains a comma-separated list of String tokens that will be treated as
        // sentence boundaries (when matched with String equality) and then discarded.
        tokensProperties.setProperty("ssplit.tokenPatternsToDiscard", "[.]");

        StanfordCoreNLP tokensPipeline = new StanfordCoreNLP(tokensProperties);

        // Para no tener un error de tipo OutOfMemoryError, es necesario cnoocer primero el número de tokens, y por
        // tanto se crea una primera pipeline para obtener los tokens y después filtrar por el número de tokens.
        Properties sentencesProperties = new Properties();
        sentencesProperties.put("annotators", "tokenize");

        StanfordCoreNLP sentencesPipeline = new StanfordCoreNLP(sentencesProperties);

        // Crea un objeto StanfordCoreNLP, con tokenización, separación de sentencias, etiquetas POS, lematización,
        // y parsing.
        Properties properties = new Properties();
        properties.put("annotators", "tokenize, ssplit, pos, lemma, parse");
        // use faster shift reduce parser
        if ("srparser".equals(processor.parserModel)) {
            properties.setProperty("parse.model", "edu/stanford/nlp/models/srparser/englishSR.ser.gz");
            System.out.println("Parser model: " + properties.getProperty("parse.model"));
        }
        // Maximum sentence length to tag. Sentences longer than this will not be tagged.
        properties.setProperty("parse.maxlen", String.valueOf(processor.maxNumTokens));
        // Whether to also store a binary version of the parse tree under BinarizedTreeAnnotation.
        properties.setProperty("parse.binaryTrees", Boolean.toString(true));
        // Number of threads to use for parsing.
        properties.setProperty("parse.nthreads", String.valueOf(processor.numThreads));

        StanfordCoreNLP pipeline = new StanfordCoreNLP(properties);


        try {
            MongoClient mongoClient = MongoDbClientHelper.getAuthenticatedMongoClient(
                    processor.host,
                    processor.port);

            MongoDatabase database = mongoClient.getDatabase(processor.dbName);
            MongoCollection<Document> collection = database.getCollection(processor.collName);

            for (int i = 1; i < 8; i = i + 3) {
                // Procesa el año de 3 en 3 meses.
                int j = i + 3;
                String endMonth = (j < 10) ? "-0" + j + "-01 00:00:00" : "-" + j + "-01 00:00:00";
                Bson query = Filters.and(
//                        Filters.eq("bug_id", 21156),
                        Filters.gte("creation_ts", processor.startYear + "-0" + i + "-01 00:00:00"),
                        Filters.lt("creation_ts", processor.startYear + endMonth),
                        Filters.ne(processor.textColumnName, "")
                );
                System.out.println(query.toString());
                processDocuments(collection, query, processor, tokensPipeline, sentencesPipeline, pipeline);
            }

            // Procesa los últimos 3 meses del año.
            Bson query = Filters.and(
//                    Filters.eq("bug_id", 21156),
                    Filters.gte("creation_ts", processor.startYear + "-10-01 00:00:00"),
                    Filters.lt("creation_ts", processor.endYear + "-01-01 00:00:00"),
                    Filters.ne(processor.textColumnName, "")
            );
            System.out.println(query.toString());
            processDocuments(collection, query, processor, tokensPipeline, sentencesPipeline, pipeline);


        } catch (MongoException e) {
            e.printStackTrace();
        }
    }
}
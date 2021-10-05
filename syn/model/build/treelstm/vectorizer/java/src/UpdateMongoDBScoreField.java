import com.mongodb.MongoException;
import com.mongodb.bulk.BulkWriteResult;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.BulkWriteOptions;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.UpdateOneModel;
import com.mongodb.client.model.WriteModel;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import org.bson.Document;
import org.bson.conversions.Bson;
import org.bson.types.ObjectId;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Recupera el título o la descripción de una incidencia existente en una colección MongoDB y crea el árbol binario de
 * constituyentes, las etiquetas gramaticales o part-of-speech, correspondientes a cada una de las palabras existentes
 * en el texto, los vectores numéricos o tag embeddings, correspondientes a cada una de las palabras existentes en el
 * texto, y almacena esta información en un fichero de texto.
 */
public class UpdateMongoDBScoreField {
    // BufferedWriter para escribir en ficheros de texto.
    private final BufferedWriter issueWriter;

    /**
     * Constructor para inicializar los valores de los atributos con los parámetros de entrada.
     *
     * @param issuePath String Ruta al fichero que almacenará las ramas asociadas a las palabras del texto tratado.
     * @throws IOException Exception Puede lanzar excepciones al escribir en ficheros de texto.
     */
    public UpdateMongoDBScoreField(
            String dbName,
            String collName,
            String issuePath,
            String startYear,
            String endYear,
            String textColumnName
    ) throws IOException {
        // Inicializa los BufferedWriter con las rutas introducidas a través de los parámetros de entrada.
        issueWriter = new BufferedWriter(new FileWriter(issuePath + "\\" + textColumnName + "_nlp_fields_" + startYear + "_" + endYear + ".sentence"));
    }

    /**
     * Cierra todos Writers abiertos.
     *
     * @throws IOException Exception Puede lanzar excepciones al escribir en ficheros de texto.
     */
    public void close() throws IOException {
        issueWriter.close();
    }

    public static void main(String[] args) throws Exception {
        Properties props = StringUtils.argsToProperties(args);

        // Argumentos de entrada.
        String dbName = props.getProperty("dbName");
        String collName = props.getProperty("collName");
        String startYear = props.getProperty("startYear");
        String endYear = props.getProperty("endYear");
        String textColumnName = "normalized_description";
        String issuePath = props.getProperty("issuePath");

        UpdateMongoDBScoreField processor = new UpdateMongoDBScoreField(
                dbName,
                collName,
                issuePath,
                startYear,
                endYear,
                textColumnName);

        // Crea un objeto StanfordCoreNLP, con tokenización, separación de sentencias, etiquetas POS, lematización,
        // y parsing.
        Properties properties = new Properties();
        properties.put("annotators", "tokenize, ssplit, pos, lemma, parse");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(properties);

        // Contador para el control del número de documentos leídos y procesados.
        AtomicInteger count = new AtomicInteger();
        AtomicLong rejectedIssuesNumber = new AtomicLong();

        AtomicInteger updateCount = new AtomicInteger();
        AtomicInteger updateRejectedCount = new AtomicInteger();

        long start = System.currentTimeMillis();

        // Capacidad inicial del buffer de escritura y procesamiento de incidencias.
        double initialCapacity = 1000.0;

        try {
            MongoClient mongoClient = MongoDbClientHelper.getAuthenticatedMongoClient(
                    "syn.altgovrd.com",
                    30017);

            MongoDatabase database = mongoClient.getDatabase(dbName);
            MongoCollection<Document> collection = database.getCollection(collName);

            Bson query = Filters.and(
                    Filters.exists("tokens", true),
                    Filters.lt("tokens", 150),
                    Filters.gte("creation_ts", startYear + "-01-01 00:00:00"),
                    Filters.lt("creation_ts", endYear + "-01-01 00:00:00"),
                    Filters.ne(textColumnName, "")
            );

            long queryCount = collection.countDocuments(query);
            System.err.println("Documentos recuperados: " + queryCount);

            // Inicializa barra de proceso en consola.
            // TODO Corregir error al ejecutar mediante subprocess en Python.
//            ETAPrinter printer = ETAPrinter.init(queryCount, System.err, true);

            // Almacena las incidencias para hacer una actualización en modo bulk.
            List<WriteModel<Document>> updateList = new ArrayList<>((int) initialCapacity);

            collection.find(query)
                    .projection(new Document("_id", 1)
                            .append("bug_id", 1)
                            .append(textColumnName, 1))
                    .noCursorTimeout(true)
                    .forEach((Consumer<Document>) (Document x) -> {
                        System.err.println("Iteracion: " + updateCount.get() + " - bug_id: " + x.get("bug_id").toString());
                        if ((x.get(textColumnName) instanceof String) && (!x.getString(textColumnName).trim().equals(""))) {
                            // read some text in the text variable
                            String text = x.getString(textColumnName);

                            // Inicializa el objeto Issue.
                            Issue issue = new Issue();

                            // Asigna el identificador.
                            issue._id = x.getObjectId("_id");
                            issue.bugId = Long.parseLong(x.get("bug_id").toString());
                            issue.leafScore = new ArrayList<>();

                            // create an empty Annotation just with the given text
                            Annotation document = new Annotation(text);

                            // NLP
                            try {
                                // Ejecuta todos los anotadores en el texto.
                                pipeline.annotate(document);

                                // these are all the sentences in this document
                                // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
                                List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

                                for (CoreMap sentence : sentences) {

                                    Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);

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
                                }

                                if (issue.attentionPOSTags > 0.0) {
                                    issue.relativeCoherence = (double) issue.absoluteCoherence / (double) issue.attentionPOSTags;
                                }

                            } catch (MongoException e) {
                                System.err.println("Iteracion: " + updateCount.get() + " - bug_id: " + issue.bugId);
                                e.printStackTrace();
                            }

                            // Acutualiza la colección Mongodb.
                            if (issue._id != null) {
//                                System.err.println("Iteracion: " + updateCount.get() + " - bug_id: " + issue.bugId);
                                updateList.add(
                                        new UpdateOneModel<>(
                                                new Document("_id", new ObjectId(issue._id.toString())), // filter
                                                new Document("$set", new Document("leaf_score", issue.leafScore)
                                                        .append("attention_pos_tags", issue.attentionPOSTags)
                                                        .append("absolute_coherence", issue.absoluteCoherence)
                                                        .append("relative_coherence", issue.relativeCoherence)
                                                )
                                        )
                                );
                                updateCount.getAndIncrement();
                            } else {
                                updateRejectedCount.getAndIncrement();
                                System.err.printf("Documentos rechazados =  %d \n", updateRejectedCount.get());
                                System.out.println(issue._id);
                            }

                            if (((updateCount.get() % initialCapacity) == 0) || (updateCount.get() == queryCount)) {
                                BulkWriteResult result = collection.bulkWrite(updateList, new BulkWriteOptions().ordered(false));
                                System.err.printf("Documentos actualizados en MongoDB: %d \n", result.getMatchedCount());
                                updateList.clear();

                                double elapsed = (System.currentTimeMillis() - start) / initialCapacity;
                                System.err.printf("Documentos procesados: %d (%.2f seg)\n", updateCount.get(), elapsed);
                            }

                        } else {
                            updateRejectedCount.getAndIncrement();
                            System.err.println("Iteracion: " + count.get() + " - bug_id rechazado: " + x.get("bug_id").toString());
                        }
                        // TODO Corregir error al ejecutar mediante subprocess en Python.
//                        printer.update(1);
                    });
        } catch (MongoException e) {
            e.printStackTrace();
        }

        long totalTimeMillis = System.currentTimeMillis() - start;
        System.err.printf("Procesados %d documentos en %.2f seg (%.1fm seg por documento)\n",
                updateCount.get(), totalTimeMillis / initialCapacity, totalTimeMillis / (double) count.get());
        System.err.println("Incidencias rechazadas: " + rejectedIssuesNumber.get());
        System.err.println("Fichero generado: " + issuePath + "\\description_nlp_fields_" + startYear + "_" + endYear + ".sentence");
        processor.close();
    }
}
import com.mongodb.MongoException;
import com.mongodb.bulk.BulkWriteResult;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.BulkWriteOptions;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.UpdateOneModel;
import com.mongodb.client.model.WriteModel;
import edu.stanford.nlp.util.StringUtils;
import org.bson.Document;
import org.bson.conversions.Bson;
import org.bson.types.ObjectId;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Elimina un campo de una colección MongoDB.
 */
public class RemoveMongoDBField {
    /**
     * Constructor para inicializar los valores de los atributos con los parámetros de entrada.
     *
     * @param dbName         String Nombre de la base de datos en la que se encuentra la colección e de la que se quiere
     *                       eliminar el campo.
     * @param collName       String Nombre de la colección en la que se encuentra el campo que se queire eliminar.
     * @param startYear      String Año de inicio utilizado para filtrar por rango de fechas los documentos para los que se
     *                       eliminará el campo.
     * @param endYear        String Año de fin utilizado para filtrar por rango de fechas los documentos para los que se
     *                       eliminará el campo.
     * @param textColumnName String Nombre del campo que se quiere eliminar de la colección MongoDB.
     */
    public RemoveMongoDBField(
            String dbName,
            String collName,
            String startYear,
            String endYear,
            String textColumnName
    ) {
        System.out.println("Clase instanciada.");
    }

    /**
     * Método main.
     *
     * @param args String[] Argumentos pasados a través de la línea de comandos.
     * @throws Exception Puede lanzar excepticones.
     */
    public static void main(String[] args) throws Exception {
        Properties props = StringUtils.argsToProperties(args);

        // Argumentos de entrada.
        String dbName = props.getProperty("dbName");
        String collName = props.getProperty("collName");
        String startYear = props.getProperty("startYear");
        String endYear = props.getProperty("endYear");
        String fieldName = props.getProperty("fieldName");

        // Contador para el control del número de documentos leídos o procesados, y rechazados.
        AtomicInteger count = new AtomicInteger();
        AtomicLong rejectedIssuesNumber = new AtomicLong();

        // Contador para el control del número de documentos actualizados y rechazados.
        AtomicInteger updateCount = new AtomicInteger();
        AtomicInteger updateRejectedCount = new AtomicInteger();

        // Almacena el momento en el que inicial la ejecución para conocer el tiempo de ejecución total.
        long start = System.currentTimeMillis();

        // Capacidad inicial del buffer de escritura y procesamiento de incidencias.
        double initialCapacity = 1000.0;

        try {
            // Cliente MongoDB.
            MongoClient mongoClient = MongoDbClientHelper.getAuthenticatedMongoClient(
                    "syn.altgovrd.com",
                    30017);

            // Base de datos y colección MongoDB en la que se encuentra el campo que se quiere borrar.
            MongoDatabase database = mongoClient.getDatabase(dbName);
            MongoCollection<Document> collection = database.getCollection(collName);

            // Consulta ejecutada para obtener los documentos que se van a actualizar.
            Bson query = Filters.and(
                    Filters.gte("creation_ts", startYear + "-01-01 00:00:00"),
                    Filters.lt("creation_ts", endYear + "-01-01 00:00:00"),
                    Filters.exists(fieldName, true)
            );

            // Información de depuración.
            System.err.println("Consulta realizada: " + query.toString());

            // Número de documentos recuperaados.
            long queryCount = collection.countDocuments(query);
            System.err.println("Documentos recuperados: " + queryCount);

            // Inicializa barra de proceso en consola.
            // TODO Corregir error al ejecutar mediante subprocess en Python.
//            ETAPrinter printer = ETAPrinter.init(queryCount, System.err, true);

            // Almacena las incidencias para hacer una actualización en modo bulk.
            List<WriteModel<Document>> updateList = new ArrayList<>((int) initialCapacity);

            // Itera sobre los documentos obtenidos.
            collection.find(query)
                    .projection(new Document("_id", 1)
                            .append("bug_id", 1)
                            .append(fieldName, 1))
                    .noCursorTimeout(true)
                    .forEach((Consumer<Document>) (Document x) -> {
                        System.err.println("Iteracion: " + updateCount.get() + " - bug_id: " + x.get("bug_id").toString());

                        // Acutualiza la colección Mongodb.
                        if (x.getObjectId("_id") != null) {
                            updateList.add(
                                    new UpdateOneModel<>(
                                            new Document("_id", new ObjectId(x.getObjectId("_id").toString())),
                                            new Document("$unset", new Document(fieldName, "")
                                            )
                                    )
                            );
                            // Actualiza el contador de elementos que se actualizarán.
                            updateCount.getAndIncrement();
                        } else {
                            // El documento no se precesará.
                            updateRejectedCount.getAndIncrement();
                            System.err.printf("Documentos rechazados =  %d \n", updateRejectedCount.get());
                            System.out.println(x.getObjectId("_id"));
                        }

                        //
                        if (((updateCount.get() % initialCapacity) == 0) || (updateCount.get() == queryCount)) {
                            BulkWriteResult result = collection.bulkWrite(updateList, new BulkWriteOptions().ordered(false));
                            System.err.printf("Documentos actualizados en MongoDB: %d \n", result.getMatchedCount());
                            updateList.clear();

                            double elapsed = (System.currentTimeMillis() - start) / initialCapacity;
                            System.err.printf("Documentos procesados: %d (%.2f seg)\n", updateCount.get(), elapsed);
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
    }
}
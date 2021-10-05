import com.mongodb.MongoClientSettings;
import com.mongodb.MongoCredential;
import com.mongodb.ServerAddress;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;

import java.util.Collections;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Clase para proporcionar un cliente MongoDB a todos las clases Java desarrolladas.
 */
public class MongoDbClientHelper {

    /**
     * Obiente un cliente MongoDB utilizado para conectar a un servidor MongoDB sin autenticación.
     *
     * @param host String Nombre del host en el que está alojado el servidor MongoDB.
     * @param port int Puerto del host en el que está alojado el servidor MongoDB.
     * @return MongoClient Cliente MongoDB.
     */
    public static MongoClient getMongoClient(String host, int port) {
        Logger mongoLogger = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);

        return MongoClients.create(
                MongoClientSettings.builder()
                        .applyToClusterSettings(builder ->
                                builder.hosts(Collections.singletonList(new ServerAddress(host, port))))
                        .build());
    }

    /**
     * Obiente un cliente MongoDB utilizado para conectar a un servidor MongoDB con autenticación.
     *
     * @param host String Nombre del host en el que está alojado el servidor MongoDB.
     * @param port int Puerto del host en el que está alojado el servidor MongoDB.
     * @return MongoClient Cliente MongoDB.
     */
    public static MongoClient getAuthenticatedMongoClient(String host, int port) {
        Logger mongoLogger = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);

        // Datos de conexión al servidor MongoDB.
        Map<String, String> env = System.getenv();

        String user = null != env.get("MONGODB_USERNAME");
        char[] password = null != env.get("MONGODB_PASSWORD").toCharArray();
        String source = null != env.get("MONGODB_AUTHENTICATION_DATABASE") ? env.get("MONGODB_AUTHENTICATION_DATABASE") : "admin";

        MongoCredential credential = MongoCredential.createCredential(user, source, password);

        return MongoClients.create(
                MongoClientSettings.builder()
                        .applyToClusterSettings(builder ->
                                builder.hosts(Collections.singletonList(new ServerAddress(host, port))))
                        .credential(credential)
                        .build());
    }
}

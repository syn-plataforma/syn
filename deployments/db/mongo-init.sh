mongo -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" <<EOF
    use $MONGO_API_DATABASE;
    db.createUser({user: '$MONGO_API_USER', pwd: '$MONGO_API_PASSWORD', roles: ["readWrite"]});
    db.createCollection('training_parameters');
    db.createCollection('users');
    use admin;
    db.createUser({user: '$MONGO_DATA_USER', pwd: '$MONGO_DATA_PASSWORD', roles: ["readAnyDatabase"]});
EOF

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db eclipse --collection clear --out=data/db/backup/eclipse/clear.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db eclipse --collection clear --file data/db/backup/eclipse/clear.json
rm -rf data/db/backups/eclipse/clear.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db eclipse --collection pairs --out=data/db/backup/eclipse/pairs.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db eclipse --collection pairs --file data/db/backup/eclipse/pairs.json
rm -rf data/db/backups/eclipse/pairs.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db netBeans --collection clear --out=data/db/backup/netBeans/clear.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db netBeans --collection clear --file data/db/backup/netBeans/clear.json
rm -rf data/db/backups/netBeans/clear.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db netBeans --collection pairs --out=data/db/backup/netBeans/pairs.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db netBeans --collection pairs --file data/db/backup/netBeans/pairs.json
rm -rf data/db/backups/netBeans/pairs.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db openOffice --collection clear --out=data/db/backup/openOffice/clear.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db openOffice --collection clear --file data/db/backup/openOffice/clear.json
rm -rf data/db/backups/openOffice/clear.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db openOffice --collection pairs --out=data/db/backup/openOffice/pairs.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db openOffice --collection pairs --file data/db/backup/openOffice/pairs.json
rm -rf data/db/backups/openOffice/pairs.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db bugzilla --collection clear --out=data/db/backup/bugzilla/clear.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db bugzilla --collection clear --file data/db/backup/bugzilla/clear.json
rm -rf data/db/backups/bugzilla/clear.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db bugzilla --collection pairs --out=data/db/backup/bugzilla/pairs.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db bugzilla --collection pairs --file data/db/backup/bugzilla/pairs.json
rm -rf data/db/backups/bugzilla/pairs.json

mongoexport --host $PROD_MONGO_DATA_HOST --port $PROD_MONGO_DATA_PORT --username $PROD_MONGO_DATA_USER --password $PROD_MONGO_DATA_PASSWORD --authenticationDatabase $PROD_MONGO_DATA_AUTH_SOURCE --authenticationMechanism $PROD_MONGO_DATA_AUTH_MECHANISM --db bugzilla --collection similar_pairs --out=data/db/backup/bugzilla/similar_pairs.json
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --db bugzilla --collection similar_pairs --file data/db/backup/bugzilla/similar_pairs.json
rm -rf data/db/backups/bugzilla/similar_pairs.json

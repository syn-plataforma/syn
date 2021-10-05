# Official base image.
FROM mongo:latest

# mongo-init.sh
COPY deployments/db/mongo-init.sh /docker-entrypoint-initdb.d/mongo-init.sh
RUN chmod +x ./docker-entrypoint-initdb.d/mongo-init.sh
RUN apt-get update && apt-get install -y dos2unix
RUN dos2unix /docker-entrypoint-initdb.d/mongo-init.sh && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

# Descarga la imagen base oficial official.
FROM python:3.8-slim

# Establece el directorio de trabajo.
WORKDIR /usr/src/app

# Establece las variables de entorno.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instala dependencias para Pycairo
RUN apt-get -y update
RUN apt-get -y install gcc libcairo2-dev pkg-config python3-dev

# Instala Java para jnius
RUN mkdir /usr/share/man/man1/; exit 0
RUN apt-get -y install default-jdk

# Instala las dependencias.
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
COPY requirements.txt /usr/src/app
RUN pip3 install -r requirements.txt

# Crea un usuario no root.
RUN addgroup --system syn && adduser --system --no-create-home --group syn

# Añade el entrypoint.sh que está en el directorio deployments
COPY deployments/app/entrypoint.sh /usr/src/app/entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Cambia el propietario del directorio de trabajo.
RUN chown -R syn:syn /usr/src/app && chmod -R 755 /usr/src/app

# Crea directorio para almacenar los datos de NLTK utilizados en el proyecto.
RUN mkdir -p /home/syn/nltk_data; exit 0
RUN chown -R syn:syn /home/syn/nltk_data && chmod -R 755 /home/syn/nltk_data

# Crea directorio para almacenar los ficheros estáticos y multi-media.
RUN mkdir -p /home/syn/web/static/; exit 0
RUN mkdir -p /home/syn/web/media/; exit 0
COPY app/static/swagger.yml /home/syn/web/static/
RUN chown -R syn:syn /home/syn/web/static

# Cambia al usuario no root.
USER syn

# Añade el código de la aplicación.
COPY definitions.py /usr/src/app
ADD ./app /usr/src/app/app
ADD ./syn /usr/src/app/syn

COPY deployments/app/gunicorn_config.py /usr/src/app/gunicorn_config.py
COPY deployments/app/check-mongodb-status.py /usr/src/app/check-mongodb-status.py

# Descarga los datos de NLTK utilizados en el proyecto.
RUN python -m nltk.downloader stopwords punkt

ENTRYPOINT ["sh", "entrypoint.sh"]
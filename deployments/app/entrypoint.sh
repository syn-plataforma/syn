#!/bin/bash
# Cambiar el final de linea a Linux
#exec gunicorn --config /usr/src/app/gunicorn_config.py app.wsgi:app
exec gunicorn --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread --log-file=- --bind 0.0.0.0:5000 app.wsgi:app
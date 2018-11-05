FROM python:3.6

ENV PYTHONPATH="/opt/obj-recognition/" \
    APP_PATH="/opt/obj-recognition/"

RUN mkdir -p ${APP_PATH}

COPY requirements.txt ${APP_PATH}

RUN apt-get update && \
    pip install -r ${APP_PATH}/requirements.txt

COPY core/ ${APP_PATH}/core/
COPY server/ ${APP_PATH}/server/
COPY test/ ${APP_PATH}/test/
COPY open-api.yaml ${APP_PATH}
COPY init.json ${APP_PATH}

CMD ["python3.6", "/opt/obj-recognition/server/app.py"]




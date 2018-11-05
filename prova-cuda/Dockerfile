FROM nvidia/cuda:9.0-runtime

RUN apt-get update && \
    apt-get -y install build-essential python-dev python-pip python-imaging wget curl



RUN pip install --upgrade pip setuptools matplotlib==2.2.0 numpy==1.14.5 gluoncv==0.4.0b20181025 mxnet-cu90==1.3.0

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

COPY yolo3 /opt/program
WORKDIR /opt/program
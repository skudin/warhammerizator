FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

LABEL org.opencontainers.image.authors="Stepan Kudin <kudin.stepan@yandex.ru>"

COPY requirements_minimal.txt /tmp/requirements.txt

ENV USER=docker
ENV GROUP=docker
ENV WORKDIR=/app
ENV PYTHONPATH=$WORKDIR
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Etc/UTC

RUN mkdir ${WORKDIR}
WORKDIR ${WORKDIR}

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils \
        curl \
        git \
        gcc \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-dev

RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt

RUN deluser ubuntu

RUN addgroup --gid 1000 ${GROUP} && \
    adduser --uid 1000 --ingroup ${GROUP} --home /home/${USER} --shell /bin/sh --disabled-password --gecos "" ${USER}

RUN USER=${USER} && \
    GROUP=${GROUP} && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: ${USER}\ngroup: ${GROUP}\n" > /etc/fixuid/config.yml

USER ${USER}:${GROUP}

ENTRYPOINT ["fixuid"]

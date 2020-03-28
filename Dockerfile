FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    curl \
    libssl-dev \
    unzip \
    htop \
    screen \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /usr/app

# TODO: Add data to .dockerignore
COPY . .

RUN pip install -r requirements.txt

ENV APP_PATH="/usr/app"
ENV PYTHONPATH "${PYTHONPATH}:${APP_PATH}"
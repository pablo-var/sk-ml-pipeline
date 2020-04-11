FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
    curl \
    libssl-dev \
    unzip \
    htop \
    screen \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

# TODO: Add data to .dockerignore
COPY . .

RUN conda env create -f environment.yaml
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

ENV APP_PATH="/usr/app"
ENV PYTHONPATH "${PYTHONPATH}:${APP_PATH}"

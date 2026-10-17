# AgrillaMoE — immagine per vast.ai Serverless (PyWorker BACKEND=llama)
#
# Multi-stage: build statica del binario (CUDA incorporato, arch multiple per
# le GPU eterogenee del marketplace) + runtime snello con supervisord (il
# server modello gira come program "llama" con log su /var/log/portal/llama.log,
# esattamente cio' che il PyWorker di vast.ai taila per la readiness).
#
# Build locale:
#   docker build -t agrillamoe-serverless -f docker/Dockerfile .
# Registro (GHCR):
#   docker tag agrillamoe-serverless ghcr.io/vagrillo/agrillamoe-serverless:latest
#   docker push ghcr.io/vagrillo/agrillamoe-serverless:latest

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src/AgrillaMoE

# bootstrap-llama.sh clona il fork llama.cpp (branch moe-expansion) in ./llama.cpp
# AGRILLA_NATIVE=0: baseline CPU portabile; arch esplicite: V100(70), A100(80),
# 3090/4090 (86/89->86), H100 (90)
RUN cd /src/AgrillaMoE \
    && ./bootstrap-llama.sh \
    && AGRILLA_BUILD_DIR=/out AGRILLA_NATIVE=0 AGRILLA_CUDA_ARCH="70;80;86;90" \
       AGRILLA_JOBS=$(nproc) ./build-linux.sh

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        git curl ca-certificates supervisor openssl procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/AgrillaMoE/dist/linux/agrillamoe /usr/local/bin/agrillamoe
COPY docker/entrypoint.sh docker/run-model.sh /usr/local/bin/
COPY docker/supervisord.conf /etc/supervisor/conf.d/llama.conf

RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/run-model.sh \
    && mkdir -p /var/log/portal /workspace/models \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# porta del backend modello verso il PyWorker (default del worker vast.ai)
ENV MODEL_SERVER_PORT=18000 \
    WORKSPACE_DIR=/workspace \
    USE_SYSTEM_PYTHON=true \
    MODELS_DIR=/workspace/models

EXPOSE 18000 3000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

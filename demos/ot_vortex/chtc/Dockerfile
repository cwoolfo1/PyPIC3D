FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
RUN python3 -m pip install \
    "jax[cuda12]==0.4.38" \
    "numpy==1.26.4" \
    "matplotlib==3.9.4" \
    "plotly==5.24.1" \
    "toml==0.10.2" \
    "tqdm==4.67.1" \
    "openpmd-api==0.16.1"

WORKDIR /opt/PyPIC3D
COPY . .
RUN chmod -R a+rX /opt/PyPIC3D
RUN python3 -m pip install --no-deps .

CMD ["python3", "-m", "PyPIC3D", "--help"]

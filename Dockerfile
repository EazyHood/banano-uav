# Imagen de produccion de banano-drone (pipeline clasico + geoespacial).
# El camino de deep learning (ultralytics/torch) NO se incluye por defecto para
# mantener la imagen ligera; instala .[deep] aparte si lo necesitas.
FROM python:3.12-slim

# libgomp1 lo necesitan scipy/scikit-image; el resto viene en las ruedas.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE config.example.yaml ./
COPY banano ./banano

RUN pip install --no-cache-dir ".[geo]"

# Carpeta de trabajo para montar datos:  docker run -v $PWD:/data ...
WORKDIR /data
ENTRYPOINT ["banano-detect"]
CMD ["--help"]

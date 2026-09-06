# syntax=docker/dockerfile:1
#
# The scoring service, and only the scoring service.
#
# Training needs LightGBM, SHAP and matplotlib; serving does not. Keeping the
# training stack out of the runtime image halves its size and removes a
# dependency surface that nothing in production would ever import.
#
#   docker build -t backstop .
#   docker run -p 8000:8000 -v $(pwd)/artifacts:/srv/artifacts:ro backstop

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/srv/src \
    OMP_NUM_THREADS=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin backstop

WORKDIR /srv

COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels

COPY src ./src
# The trained artifact travels with the image so the container is a complete,
# reproducible unit: what was tested is what runs. Mount over it to serve a
# different model without rebuilding.
COPY artifacts/models ./artifacts/models

RUN chown -R backstop:backstop /srv
USER backstop

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# One worker: the latency and score histograms on /metrics are process-local,
# and the model is small enough that a replica costs less than shared state.
CMD ["uvicorn", "backstop.serving.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--no-server-header"]

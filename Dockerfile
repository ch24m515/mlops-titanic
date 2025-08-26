# Dockerfile
FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

# Bash (Spark scripts), ps (procps), tini, certificates, and Java (JRE)
# Try Java 17 first; if not available, fall back to distro default JRE.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends bash procps tini ca-certificates; \
    (apt-get install -y --no-install-recommends openjdk-17-jre-headless) \
    || (apt-get install -y --no-install-recommends default-jre-headless); \
    rm -rf /var/lib/apt/lists/*

# Detect JAVA_HOME dynamically (don’t hardcode paths)
RUN JAVA_BIN="$(readlink -f "$(which java)")"; \
    JAVA_HOME="$(dirname "$(dirname "$JAVA_BIN")")"; \
    echo "export JAVA_HOME=$JAVA_HOME" > /etc/profile.d/java.sh

# Python deps
RUN pip install --no-cache-dir pyspark==3.5.1 mlflow fastapi uvicorn

# Spark networking niceties for Docker/WSL
ENV SPARK_DRIVER_BIND_ADDRESS=127.0.0.1 \
    SPARK_LOCAL_IP=127.0.0.1 \
    SPARK_LOCAL_HOSTNAME=localhost \
    PYSPARK_PYTHON=python \
    PYSPARK_DRIVER_PYTHON=python

WORKDIR /app
COPY . /app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

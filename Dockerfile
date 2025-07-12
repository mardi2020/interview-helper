FROM python:3.12-slim AS build

ARG TARGETARCH

WORKDIR /app

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake g++ && \
      rm -rf /var/lib/apt/lists/* ; \
    else \
      echo "Non-ARM architecture detected, skipping build tools installation"; \
    fi

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install numpy \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]

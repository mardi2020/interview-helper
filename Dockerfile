FROM python:3.12-slim

ARG TARGETARCH

WORKDIR /app

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      apt-get update && apt-get install -y build-essential cmake g++ && rm -rf /var/lib/apt/lists/* ; \
    else \
      echo "Non-ARM architecture detected, skipping build tools installation"; \
    fi

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install numpy \
    && pip install -r requirements.txt

COPY . .
COPY .env .env

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
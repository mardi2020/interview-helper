FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
COPY .env .env

EXPOSE 8501

CMD ["sh", "-c", "streamlit run main.py"]
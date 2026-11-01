
FROM python:3.11-slim

WORKDIR /app

# Instal dependency khusus model serving
COPY requirements-serving.txt .

RUN pip install --no-cache-dir -r requirements-serving.txt

# Salin aplikasi dan artefak model Champion
COPY app.py .
COPY models/champion/ ./models/champion/

# Port FastAPI di dalam container
EXPOSE 8000

# Jalankan API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

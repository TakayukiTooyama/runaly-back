FROM python:3.8.18-slim-bullseye

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
  apt-get -y install --no-install-recommends libgl1-mesa-glx libglib2.0-0 libglib2.0-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

COPY ./app /app

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

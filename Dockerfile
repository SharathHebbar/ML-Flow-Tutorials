FROM python:3.9.18-slim

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["python", "rf.py"; "mlflow" "ui"]

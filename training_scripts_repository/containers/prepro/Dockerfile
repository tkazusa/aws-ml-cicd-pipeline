FROM python:3.7-buster

ADD . /src
WORKDIR /src

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "/src/preprocess.py"]
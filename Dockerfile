FROM python:3.10

WORKDIR /kc_recsys

COPY ./requirements.txt /kc_recsys/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /kc_recsys/requirements.txt

COPY ./app /kc_recsys/app

WORKDIR app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

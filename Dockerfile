FROM python:3.11.5


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./initialize.py /code/initialize.py
RUN python3 /code/initialize.py

COPY . .


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]


FROM python:3.7
RUN mkdir /nnserver/
COPY docker/conf/bin /nnserver/bin
COPY . /nnserver
RUN pip install --upgrade pip && pip install -r /nnserver/nnserver/requirements.txt
WORKDIR /nnserver
RUN python setup.py develop
WORKDIR /nnserver/nnserver
CMD gunicorn --bind 0.0.0.0:5000 main:app  --workers 3


FROM python:3.7
EXPOSE 5005
RUN mkdir /nnserver/
COPY docker/conf/bin /nnserver/bin
COPY . /nnserver
RUN pip install --upgrade pip && pip install -r /nnserver/nnserver/requirements.txt
WORKDIR /nnserver
RUN python setup.py develop
WORKDIR /nnserver/nnserver
CMD gunicorn --bind 0.0.0.0:5005 main:app --workers 3 --timeout 300


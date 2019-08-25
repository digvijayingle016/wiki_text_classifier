FROM joelogan/keras-tensorflow-flask-uwsgi-nginx-docker

# ------------------------------------------------------- #
#                Import App 
# ------------------------------------------------------- #
COPY ./app /app
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt 
RUN pip install -r /tmp/requirements.txt
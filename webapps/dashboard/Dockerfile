FROM ubuntu:24.04 AS build

RUN apt-get update  && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-setuptools python3-wheel \
                                                curl build-essential python3-dev

# build wheels for all requirements
WORKDIR /wheels

COPY requirements.txt requirements.txt
RUN pip3 wheel -r requirements.txt

FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-setuptools python3-wheel \
                                            python3-dev sqlite3 python3-venv wget curl git

# install all the python dependencies
COPY --from=build /wheels /tmp/wheels

# create a venv and use it
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip3 install /tmp/wheels/*.whl

ENV PYTHONIOENCODING=utf8

WORKDIR /app

# copy the application sources and the uwsgi configuration
COPY uwsgi.ini manage.py /app/
COPY perf /app/perf
COPY webapp /app/webapp
COPY templates /app/templates

# collect all the static files for serving over http
RUN python3 manage.py collectstatic --noinput

EXPOSE 8080

USER ubuntu

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhosté8080/health/ || exit 1

CMD ["uwsgi", "uwsgi.ini"]

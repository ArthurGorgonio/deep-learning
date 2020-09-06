FROM python:3.8.5-alpine

LABEL maintainer="Arthur Gorgonio"

RUN echo -e 'http://nl.alpinelinux.org/alpine/edge/community\n\
  http://nl.alpinelinux.org/alpine/edge/main\n\
  http://nl.alpinelinux.org/alpine/edge/testing\n\
  http://dl-cdn.alpinelinux.org/alpine/edge/community'\
  >> /etc/apk/repositories && \
  apk upgrade && \
  apk add --update --no-cache \
  build-base \
  freetype-dev \
  gcc \
  gfortran \
  jpeg-dev \
  musl-dev \
  openblas-dev && \
  pip install --no-cache-dir --upgrade pip requests && \
  pip install --no-cache-dir numpy Cython && \
  ln -s /usr/include/locate.h /usr/include/xlocate.h

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY start.sh /

RUN chmod +x /start.sh

COPY src/ .

CMD ["/bin/sh", "/start.sh"]

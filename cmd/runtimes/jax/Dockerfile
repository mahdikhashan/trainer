FROM python:3.12

RUN apt-get update && apt install -y --no-install-recommends \
    cmake \ 
    build-essential \
    git \
    libstdc++6

RUN git clone https://github.com/pytorch/gloo.git
RUN cd gloo \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make \
    && make install

RUN ls -lha /usr/local/lib | grep gloo
RUN ls -lha /usr/local/include/gloo

RUN cd gloo && git rev-parse HEAD

# 2025-05-31 17:10:47 -rw-r--r--  1 root root 1.6M May 31 15:01 libgloo.a
CMD ["sh", "-c", "ls -lha /usr/local/lib | grep gloo"]

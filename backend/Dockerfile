FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV POETRY_VERSION=1.1.11
ENV ENVIRONMENT=DEV
# Installing packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get install -y ffmpeg

#RUN : \
#    && apt-get update \
#    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#        software-properties-common \
#    && add-apt-repository -y ppa:deadsnakes \
#    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#        python3.8-venv \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/* \
#    && :

RUN apt-get update

RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt-get install python3.8 -y
RUN apt-get install python3.8-dev -y
RUN apt-get install python3-pip -y

WORKDIR /backend
COPY  ./requirements.txt  /backend/
RUN python3.8 -m pip install -r requirements.txt
# RUN python3.8 -m pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu101


COPY . /backend

# Start app
CMD ["python3.8", "main.py"]

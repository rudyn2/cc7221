FROM anibali/pytorch:1.5.0-cuda10.2
USER root

# RUN conda env create -f environment.yml
# RUN apt-get install -y gnupg2
# RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic main" >> /etc/apt/sources.list
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
# RUN apt-get update

RUN mkdir /home/cc7221
WORKDIR /home/cc7221
# RUN apt-get update
RUN pip install wandb
RUN pip install torchvision==0.9.1

# Copy code
COPY *.py /home/cc7221/

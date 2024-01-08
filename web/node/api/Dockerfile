FROM ubuntu:20.04
RUN apt update
ADD ./packages.txt /packages.txt
ADD ./requirements.txt /requirements.txt
RUN apt install -y $(grep -vE "^\s*#" /packages.txt  | tr "\n" " ")
RUN pip install -r /requirements.txt

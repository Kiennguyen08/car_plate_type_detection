FROM ubuntu:18.04
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get install -y libgl1-mesa-dev
COPY . /project
WORKDIR /project 
RUN pip3 install --upgrade pip
RUN pip3 install -r setup.txt
RUN pip3 install --no-cache-dir tensorflow==2.2.0
CMD ["python3","server.py"]

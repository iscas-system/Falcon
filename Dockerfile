FROM frankwu2016/centos-botorch:latest
    
RUN mkdir -p /home/falcon/data
RUN mkdir -p /home/falcon/src
ADD ./data /home/falcon/data
ADD ./scripts/* /usr/bin/
ADD ./src /home/falcon/src

WORKDIR /home/falcon

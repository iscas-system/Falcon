FROM centos:centos7.9.2009
ENV PATH /usr/local/bin:$PATH
ENV PATH /root/.local/bin:$PATH
    
ADD ./REQUIREMENTS /tmp
RUN yum install python3 python3-devel python3-pip git -y
RUN git config --global https.proxy 'https://133.133.133.111:4780'
RUN git config --global http.proxy 'http://133.133.133.111:4780'
RUN python3.6 -m pip install --user --upgrade pip
RUN python3.6 -m pip install --user -r /tmp/REQUIREMENTS
RUN yum clean all
RUN rm -fr /tmp/*

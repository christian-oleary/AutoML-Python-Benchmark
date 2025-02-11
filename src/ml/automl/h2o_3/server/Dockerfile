########################################################################
# Dockerfile for Oracle JDK 8 on Ubuntu 16.04
########################################################################

# pull base image
FROM ubuntu:16.04

# add a post-invoke hook to dpkg which deletes cached deb files
# update the sources.list
# update/dist-upgrade
# clear the caches
RUN echo 'DPkg::Post-Invoke {"/bin/rm -f /var/cache/apt/archives/*.deb || true";};' | tee /etc/apt/apt.conf.d/no-cache && \
    echo "deb http://mirror.math.princeton.edu/pub/ubuntu xenial main universe" >> /etc/apt/sources.list && \
    apt-get update -q -y && \
    apt-get dist-upgrade -y && \
    apt-get clean && \
    rm -rf /var/cache/apt/* && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget unzip openjdk-8-jdk python-pip python-sklearn python-pandas python-numpy python-matplotlib software-properties-common python-software-properties && \
    apt-get clean

# Fetch h2o latest_stable
# RUN wget http://h2o-release.s3.amazonaws.com/h2o/latest_stable -q -O latest
# # "latest" contains the URL of the latest stable H2O build
# RUN wget -i latest -O /opt/h2o.zip
# RUN wget -q -i https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/6/h2o-3.46.0.6.zip -O /opt/h2o.zip

ADD h2o.zip /opt/h2o.zip

RUN unzip -d /opt /opt/h2o.zip && \
    rm /opt/h2o.zip && \
    cd /opt && \
    cd `find . -name 'h2o.jar' | sed 's/.\///;s/\/h2o.jar//g'` && \
    cp h2o.jar /opt && \
    printf '!/bin/bash\ncd /home/h2o\n./start-h2o-docker.sh\n' > /start-h2o-docker.sh && \
    chmod +x /start-h2o-docker.sh

# RUN useradd -m -c "h2o.ai" h2o
# USER h2o

# Get Content
RUN cd && \
    wget https://raw.githubusercontent.com/h2oai/h2o-3/master/docker/start-h2o-docker.sh && \
    chmod +x start-h2o-docker.sh

# Link currently not working
# RUN wget -q http://s3.amazonaws.com/h2o-training/mnist/train.csv.gz
# RUN gunzip train.csv.gz

# Define the working directory
# WORKDIR /home/h2o

RUN apt-get -y update
RUN mkdir -p /usr/share/man/man1/
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y openjdk-8-jre
RUN update-alternatives --config java
RUN update-alternatives --config javac

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -y install default-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN java -version

EXPOSE 54321
EXPOSE 54322

ENTRYPOINT ["java", "-Xmx4g", "-jar", "/opt/h2o.jar"]

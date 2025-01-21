# Description: Dockerfile for H2O-3
# Adapted from: https://github.com/h2oai/h2o-3/blob/master/Dockerfile

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
RUN wget http://h2o-release.s3.amazonaws.com/h2o/latest_stable -O latest && \
    wget -i latest -O /opt/h2o.zip && \
    unzip -d /opt /opt/h2o.zip && \
    rm /opt/h2o.zip

# Install h2o
RUN cd /opt && \
    cd `find . -name 'h2o.jar' | sed 's/.\///;s/\/h2o.jar//g'`




# > [ 5/11] RUN cp h2o.jar /opt:
#     0.284 cp: cannot stat 'h2o.jar': No such file or directory
#     ------
#     Dockerfile:30
#     --------------------
#       28 |         cd `find . -name 'h2o.jar' | sed 's/.\///;s/\/h2o.jar//g'`
#       29 |
#       30 | >>> RUN cp h2o.jar /opt
#       31 |     RUN /usr/bin/pip install `find . -name "*.whl"`
#       32 |     RUN printf '!/bin/bash\ncd /home/h2o\n./start-h2o-docker.sh\n' > /start-h2o-docker.sh
#     --------------------
#     ERROR: failed to solve: process "/bin/sh -c cp h2o.jar /opt" did not complete successfully: exit code: 1
RUN cp h2o.jar /opt









RUN /usr/bin/pip install `find . -name "*.whl"`
RUN printf '!/bin/bash\ncd /home/h2o\n./start-h2o-docker.sh\n' > /start-h2o-docker.sh

# Make the script executable
RUN chmod +x /start-h2o-docker.sh

# Create new user
RUN useradd -m -c "h2o.ai" h2o
USER h2o

# Get Content
RUN cd && \
    wget https://raw.githubusercontent.com/h2oai/h2o-3/master/docker/start-h2o-docker.sh && \
    chmod +x start-h2o-docker.sh && \
    wget http://s3.amazonaws.com/h2o-training/mnist/train.csv.gz && \
    gunzip train.csv.gz

# Define working directory
WORKDIR /home/h2o

EXPOSE 54321
EXPOSE 54322

#ENTRYPOINT ["java", "-Xmx4g", "-jar", "/opt/h2o.jar"]

CMD ["/bin/bash"]

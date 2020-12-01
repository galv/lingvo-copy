ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Daniel Galvez"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        dirmngr \
        git \
        gpg-agent \
        less \
        libboost-all-dev \
        libeigen3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libbz2-dev \
        liblzma-dev \
        libpng-dev \
        libzmq3-dev \
        libsox-fmt-mp3 \
        lsof \
        pkg-config \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN export CLOUDSDK_CORE_DISABLE_PROMPTS=1 CLOUDSDK_INSTALL_DIR=/install \
    && curl https://sdk.cloud.google.com | bash

# TODO: Set the configurations in launch_pyspark_notebook.sh here as well.
RUN /bin/bash -c "source /install/google-cloud-sdk/path.bash.inc && \
    curl -O https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz \
    && mkdir -p /install/spark \
    && tar zxf spark-3.0.0-bin-hadoop2.7.tgz -C /install/spark --strip-components=1 \
    && gsutil cp gs://hadoop-lib/gcs/gcs-connector-hadoop2-2.1.6.jar /install/spark/jars"

ENV SPARK_HOME="/install/spark"
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
ENV SPARK_CONF_DIR="$SPARK_HOME/conf"

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /install/miniconda3

COPY environment2.yml /install/environment2.yml
ENV PATH="/install/miniconda3/bin/:${PATH}"
RUN conda env create -f /install/environment2.yml

RUN curl -L -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.4/bazelisk-linux-amd64 \
     && chmod a+x /usr/local/bin/bazel \
     && curl -L -o /usr/local/bin/ctpu https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu \
     && chmod a+x /usr/local/bin/ctpu

RUN conda init bash \
    && echo "conda init bash; conda activate 100k-hours-lingvo-3" >> $HOME/.bashrc

RUN curl -L -o /install/spark/jars/spark-tfrecord_2.12-0.3.0.jar \
    https://search.maven.org/remotecontent?filepath=com/linkedin/sparktfrecord/spark-tfrecord_2.12/0.3.0/spark-tfrecord_2.12-0.3.0.jar

RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.sh \
    && chmod +x cmake-3.19.1-Linux-x86_64.sh \
    && ./cmake-3.19.1-Linux-x86_64.sh --exclude-subdir --skip-license --prefix=/usr/local/

# https://100k-hours.slack.com/archives/CLQAYP797/p1605036830303800
# https://100k-hours.slack.com/archives/CLQAYP797/p1605045990306700?thread_ts=1605039855.304800&cid=CLQAYP797
# TODO: Download a particular version of kenlm, not just the latest one.
RUN mkdir -p /install/kenlm/ \
    && curl -L -o /install/kenlm/kenlm.tar.gz https://kheafield.com/code/kenlm.tar.gz \
    && cd /install/kenlm/ \
    && tar zxf kenlm.tar.gz --strip-components=1 \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j $(nproc)

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888
EXPOSE 8880

WORKDIR "/development/lingvo-source"

CMD ["/bin/bash", "-c"]

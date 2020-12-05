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
        emacs \
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
        lsof \
        pkg-config \
        rename \
        rsync \
        unzip \
        vim \
        wget \
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

# This is probably not necessary. I initially believed I needed to run autoconf, but this is no longer the case.
RUN apt-get update && apt-get install -y --no-install-recommends autoconf autotools-dev automake libtool

RUN mkdir -p /install/mad/ \
    && curl -L -o /install/mad/mad.tar.gz https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz \
    && cd /install/mad \
    && tar zxf mad.tar.gz --strip-components=1 \
    && sed -i '/-fforce-mem/d' ./configure \
    && ./configure --prefix=/usr --disable-debugging --enable-fpm=64bit \
    && make install

RUN mkdir -p /install/lame/ \
    && curl -L -o /install/lame/lame.tar.gz https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz \
    && cd /install/lame \
    && tar zxf lame.tar.gz --strip-components=1 \
    && ./configure --prefix=/usr \
    && make install

RUN mkdir -p /install/sox/ \
    && curl -L -o /install/sox/sox.tar.gz https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz \
    && cd /install/sox \
    && tar zxf sox.tar.gz --strip-components=1 \
    && ./configure --prefix=/usr \
    && make install

COPY third_party/DeepSpeech/ /install/mozilla-DeepSpeech/

RUN git clone https://github.com/mozilla/tensorflow.git /install/tensorflow-deepspeech-fork \
    && cd /install/tensorflow-deepspeech-fork \
    && git checkout 4e0e823493f581df7634c08235698741c4c66207 \
    && export TFDIR=/install/tensorflow-deepspeech-fork \
    && export USE_BAZEL_VERSION=0.24.1 \
    && ln -s /install/mozilla-DeepSpeech/native_client ./ \
    && conda run -n 100k-hours-lingvo-3 ./configure \
    && conda run -n 100k-hours-lingvo-3 bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx512f //native_client:libdeepspeech.so \
    && cd /install/mozilla-DeepSpeech/native_client \
    && conda run -n 100k-hours-lingvo-3 make deepspeech \
    && cd python \
    && conda run -n 100k-hours-lingvo-3 make bindings \
    && conda run -n 100k-hours-lingvo-3 pip install dist/deepspeech*

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888
EXPOSE 8880

WORKDIR "/development/lingvo-source"

CMD ["/bin/bash", "-c"]

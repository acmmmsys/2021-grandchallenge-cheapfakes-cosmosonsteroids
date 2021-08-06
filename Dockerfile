FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Setup Environment Variables
ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

ENV COSMOS_BASE_DIR="/opt/COSMOS" \
    COSMOS_DATA_DIR="/mmsys21cheapfakes" \
    COSMOS_IOU="0.25" \
    COSMOS_RECT_OPTIM="1"

# Copy Dependencies
COPY requirements.txt /
COPY detectron2 /detectron2
COPY detectron2_changes /detectron2_changes

# Prepare Environment (1)
RUN mkdir -p /opt/COSMOS/models_final

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-dev python3-pip python3-opencv \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Prepare Python Dependencies
RUN python3 -m pip install --upgrade pip && \
    pip3 install cython numpy setuptools && \
    pip3 install -r /requirements.txt

# Patch and Install Detectron
RUN cd /detectron2/ && \
    patch -p1 < /detectron2_changes/0001-detectron2-mod.patch && \
    cd / && python3 -m pip install -e detectron2

# Fix PyCocoTools
RUN pip3 uninstall -y pycocotools && \
    pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Download spaCy
RUN python3 -m spacy download en && \
    python3 -m spacy download en_core_web_sm

# Copy Source
COPY . /opt/COSMOS

# Start the code
ENTRYPOINT []
CMD ["/opt/COSMOS/start.sh"]

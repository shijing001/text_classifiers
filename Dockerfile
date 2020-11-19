FROM nvidia/cudagl:9.2-devel-ubuntu16.04

# Install prerequisites
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python-setuptools python-dev freeglut3 freeglut3-dev libgtk2.0-dev pkg-config
RUN apt-get update && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN apt-get install vim -y 


# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN easy_install pip
# RUN pip install torch==1.1.0 torchvision==0.4.0 
RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
RUN pip install pandas networkx
RUN pip install tensorboardX==1.6 scikit-learn==0.20

# ENV TORCH_CUDA_ARCH_LIST=Volta;Turing;Kepler+Tesla
ENV TORCH_CUDA_ARCH_LIST="7.0"
RUN apt-get install python3-dev -y

# RUN git clone https://github.com/NVIDIA/apex.git \
#  && cd apex \
#  && git checkout ccffa71cc566ab40e3be59743b8a10c9efc1b845 \
#  && python setup.py install --cuda_ext --cpp_ext  


# Q-GAN
RUN pip install tqdm  
RUN pip install azure    
RUN pip install nltk
RUN pip install requests
RUN pip install h5py
RUN pip install scipy
RUN pip install Pillow==6.1

WORKDIR /workspace

ADD . /workspace/transformers
# RUN rm -r /workspace/transformers/tmp/*
RUN cd /workspace/transformers \
 && pip install .

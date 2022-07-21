FROM pytorchlightning/pytorch_lightning

# TODO [noasc] So should we inherit from meod-momo or mx-conda?
#FROM artifactory.sddc.mobileye.com/objd-docker-local/meod-momo:latest

# TODO [noasc] what is it?
#RUN echo me.docker:x:48214:1000::/home/me.docker:/bin/bash >> /etc/passwd
# docker run --rm -it -v `pwd`:/galk/ensemble --gpus="device=3" --shm-size 64g --ipc host dens

# TODO [noasc] why installing the packages in the base env and then creating a new env named 'test_env'?
RUN pip install wandb
RUN pip install ipdb
ENV WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7
ENV WANDB_DIR=/ml/imagenet/wandb
ENV WANDB_CONFIG_DIR=/ml/imagenet/wandb/.config
WORKDIR /ml/imagenet/

RUN pip install hydra-core --upgrade
RUN pip install numpy==1.21.0

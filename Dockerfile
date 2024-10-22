# source .env && docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e "WANDB_API_KEY=$WANDB_API_KEY" --rm -v $PWD/data:/data -v $PWD/logs:/logs $(docker build . -q)
FROM nvcr.io/nvidia/pytorch:24.07-py3

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --user
COPY src src
WORKDIR src
COPY run.sh run.sh
CMD /bin/bash run.sh

FROM python:3.11.5-slim
 
WORKDIR /usr/src

RUN apt-get update && \
    apt-get install nano && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get install -y python3-opencv

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt 

RUN adduser --disabled-password markbotros

USER markbotros

COPY --chown=markbotros:markbotros --chmod=750 . .
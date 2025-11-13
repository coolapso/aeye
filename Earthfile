VERSION 0.8
IMPORT --allow-privileged github.com/coolapso/dry//python AS common
ARG --global name = aeye
ARG --global tag = dev
ARG --global registry = ghcr.io/coolapso
ARG --global model = models/arcticskies_efficientnet_86.keras
ARG --global source = rtsps://10.20.0.1:7441/nUtRlxvxXC81ul5i?enableSrtp


venv:
  DO common+VENV

deps:
  DO common+DEPS

init:
  DO common+INIT

core: 
  FROM DOCKERFILE .
  SAVE IMAGE --push $registry/$name:$tag-base
  SAVE IMAGE --push $registry/$name:latest-base

build:
  FROM +core
  COPY models/ models/
  SAVE IMAGE --push $registry/$name:$tag
  SAVE IMAGE --push $registry/$name:latest

build-all:
  BUILD +core
  BUILD +build

test:
  LOCALLY
  WITH DOCKER
    RUN docker run \
      -p 8000:8000 \
      -e MODE="testing" \
      -e VIDEO_SOURCE="$source" \
      -e MODEL_NAME="$model" "$registry/$name:$tag"
  END

run:
  LOCALLY
    RUN QT_QPA_PLATFORM=xcb \
      VIDEO_SOURCE="$source" \
      MODEL_NAME="$model" \
      python src/aeye.py
      
run-testing:
  LOCALLY
    RUN QT_QPA_PLATFORM=xcb \
      MODE=testing \
      VIDEO_SOURCE="$source" \
      MODEL_NAME="$model" \
      python src/aeye.py

run-classify:
  LOCALLY
    RUN QT_QPA_PLATFORM=xcb \
        MODE=classify \
        VIDEO_SOURCE="$source" \
        FRAMES_OUTPUT_ROOT_DIR="test-dataset" \
        python src/aeye.py

run-dynamic:
  LOCALLY
    RUN QT_QPA_PLATFORM=xcb \
        MODE=dynamic \
        VERBOSE=True \
        VIDEO_SOURCE="$source" \
        FRAMES_OUTPUT_ROOT_DIR="test-dataset" \
        python src/aeye.py

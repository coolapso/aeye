FROM python:3.12.2

ENV TF_CPP_MIN_LOG_LEVEL=3
EXPOSE 8000
ENV TESTING="false"
ENV VIDEO_SOURCE=""

RUN apt update && apt install -y libopencv-dev

WORKDIR /app

COPY src/ src/
COPY requirements.txt .

RUN pip install -r requirements.txt
ENTRYPOINT ["python", "src/aeye.py"]

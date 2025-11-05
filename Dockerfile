FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda11.8-cudnn8.9
COPY . /workspace
WORKDIR /workspace
RUN apt-get update && apt-get install -y libgl1 && \
    apt-get install -y libglib2.0-0 && \
    pip install -r requirements.txt
CMD ["python","api_server.py"]
FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/python:3.10

WORKDIR /home/user/app

RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

ENV MODELSCOPE_API_KEY=""
ENV DASHSCOPE_API_KEY=""

ENTRYPOINT ["python", "-u", "app.py"]

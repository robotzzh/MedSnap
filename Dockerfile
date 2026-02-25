FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/python:3.10

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir PyMuPDF -i https://mirrors.aliyun.com/pypi/simple/ || echo "PyMuPDF install skipped, PDF support disabled"

COPY . .

ENTRYPOINT ["python", "-u", "app.py"]

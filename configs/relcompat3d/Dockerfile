FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.lock.txt

COPY src/relcompat3d/ /workspace/src/relcompat3d/

ENTRYPOINT ["python"]
CMD ["--version"]

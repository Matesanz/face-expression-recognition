ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install OpenCV
RUN apt update && apt install libopencv-dev -y

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

# Copy source code
COPY app/ /app/
COPY assets/ /assets/
WORKDIR /app/

# Run Streamlit App
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]

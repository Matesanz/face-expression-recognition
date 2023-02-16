ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install OpenCV
RUN apt update && apt install libopencv-dev -y

# Install Poetry
ARG POETRY="1.2.2"
RUN pip install poetry==${POETRY}

# Install dependencies
COPY pyproject.toml /
RUN poetry config virtualenvs.create false \
   && poetry install --no-interaction --no-ansi --no-root

# Copy source code
COPY app/ /app/
COPY assets/ /assets/
RUN pip install -e .

# Run Streamlit App
WORKDIR /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]

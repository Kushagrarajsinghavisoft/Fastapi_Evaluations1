FROM python:3.11

WORKDIR /Weekly_Assignment_8

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -

ENV PATH="/etc/poetry/bin:$PATH"

COPY ../poetry.lock /Weekly_Assignment_8/
COPY ../pyproject.toml /Weekly_Assignment_8/
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-ansi 
COPY .. .


CMD ["uvicorn", "ml_deployment.fastapi_application:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["sleep", "infinity"]
FROM python:3.10.13-slim

RUN pip install -U pip 
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy
RUN pip install pandas

COPY [ "homework_notebook.py", "model.bin", "./" ]

ENTRYPOINT [ "python", "./homework_notebook.py"]
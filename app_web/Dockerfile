# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    python3-opencv

WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
RUN echo abc2
COPY ../ .
CMD ["flask", "run"]
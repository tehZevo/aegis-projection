FROM tensorflow/tensorflow

WORKDIR /app

COPY requirements.txt .
RUN apt update -y
RUN apt install git -y
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD [ "python", "-u", "main.py" ]

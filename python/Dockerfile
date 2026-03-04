# Do not change the following line. It specifies the base image which
# will be downloaded when you build your image.
FROM pklehre/ec2025-lab2

RUN apt-get update
RUN apt-get -y install python3-numpy

COPY ./common /bin/common
COPY ./questions /bin/questions
COPY ./main.py /bin/main.py

CMD ["-username", "abc123", "-submission", "python3 /bin/main.py"]
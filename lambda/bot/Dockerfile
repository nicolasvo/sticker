FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc git mesa-libGL
RUN pip install python-telegram-bot==13.10 boto3 torch torchvision numpy opencv-python Pillow==9.5.0 
RUN pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git 

# COPY app.py user.py sticker.py image.py dynamodb.py emojis.py ${LAMBDA_TASK_ROOT}
COPY app.py sticker.py dynamodb.py ${LAMBDA_TASK_ROOT}
ENV NUMBA_CACHE_DIR=/tmp

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]

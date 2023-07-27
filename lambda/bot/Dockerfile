FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc git mesa-libGL
RUN pip install python-telegram-bot requests boto3 numpy opencv-python Pillow==9.5.0

# COPY app.py user.py sticker.py image.py dynamodb.py emojis.py ${LAMBDA_TASK_ROOT}
COPY app.py sticker.py image.py dynamodb.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]

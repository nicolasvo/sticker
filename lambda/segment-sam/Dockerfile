FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN yum install -y gcc git mesa-libGL
RUN pip install torch torchvision numpy opencv-python Pillow==9.5.0
RUN pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

# COPY app.py user.py sticker.py image.py dynamodb.py emojis.py ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]

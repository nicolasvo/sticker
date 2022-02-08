FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt .
COPY app.py emojis.py ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

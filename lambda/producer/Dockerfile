FROM public.ecr.aws/lambda/python:3.9

COPY app.py emojis.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt .
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

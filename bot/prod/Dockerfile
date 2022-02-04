FROM sticker-base:latest

COPY xception_pascalvoc.pb ${LAMBDA_TASK_ROOT}
COPY app.py alter_background.py user.py sticker.py image.py emojis.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

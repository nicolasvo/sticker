FROM sticker-base:latest

COPY app.py alter_background.py user.py sticker.py emojis.py xception_pascalvoc.pb ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

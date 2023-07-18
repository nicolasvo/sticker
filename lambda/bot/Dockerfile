FROM sticker-sam:base

COPY app.py user.py sticker.py image.py emojis.py ${LAMBDA_TASK_ROOT}
ENV NUMBA_CACHE_DIR=/tmp

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

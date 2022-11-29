FROM sticker-base:latest

COPY xception_pascalvoc.pb modnet_photographic_portrait_matting_opset9.onnx ${LAMBDA_TASK_ROOT}
COPY u2net.onnx /root/.u2net/
COPY app.py alter_background.py deeplab.py user.py sticker.py image.py models.py emojis.py ${LAMBDA_TASK_ROOT}
ENV NUMBA_CACHE_DIR=/tmp

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

FROM python:3.10-slim-bookworm

ARG BUILD_VERSION
ARG BUILD_ARCH

WORKDIR /app

COPY python/ ./python/
COPY wyoming/ ./wyoming/
COPY models/ ./models/
COPY run.sh /run.sh

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        onnxruntime==1.16.3 \
        "wyoming>=1.8,<2" \
    && chmod 0755 /run.sh

LABEL org.opencontainers.image.source="https://github.com/houtacheng/onnx-wakeword" \
      org.opencontainers.image.description="Voicute Wyoming wake-word add-on for Home Assistant Green" \
      org.opencontainers.image.licenses="MIT" \
      io.hass.version="${BUILD_VERSION}" \
      io.hass.type="app" \
      io.hass.arch="${BUILD_ARCH}"

EXPOSE 10400

CMD ["/run.sh"]

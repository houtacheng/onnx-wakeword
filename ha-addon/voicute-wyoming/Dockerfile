FROM voicute/voicute-wyoming:1.0.0

# The upstream image contains the inference engine and demo models, but its
# server uses a non-standard frame format. Install the reference Wyoming
# implementation and overlay the Home Assistant compatible server maintained
# in this repository.
RUN pip install --no-cache-dir "wyoming>=1.8,<2"
ADD https://raw.githubusercontent.com/houtacheng/onnx-wakeword/main/wyoming/wyoming_voicute_ha.py /app/wyoming/wyoming_voicute_ha.py

# Add-on entrypoint: reads Home Assistant options from /data/options.json and
# launches the Wyoming service. Plain Python (no bash/bashio) — the base image
# is python:3.10-slim, which has neither.
COPY entrypoint.py /entrypoint.py
RUN chmod a+x /entrypoint.py

# Override the base image's ENTRYPOINT (python wyoming/wyoming_voicute.py),
# otherwise CMD below would be appended to it as an argument.
ENTRYPOINT ["python", "/entrypoint.py"]
CMD []

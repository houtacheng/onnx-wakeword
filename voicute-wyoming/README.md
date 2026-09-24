# Voicute Wake Word add-on

This add-on runs the repository's bundled Voicute ONNX wake-word models as a
Wyoming service on TCP port 10400. It is packaged for the `aarch64`
architecture used by Home Assistant Green.

After installation, add a Wyoming Protocol integration in Home Assistant with
the Home Assistant Green IP address as the host and `10400` as the port.

# Keepout - OpenCV AI Competition 2026
#
# Build from the REPOSITORY ROOT, not from this directory, because the image
# needs the two shared workspace packages:
#
#     docker build -f products/keepout/Dockerfile -t keepout .
#     infra/ecr.sh keepout --context . --dockerfile products/keepout/Dockerfile
#
# linux/amd64 on purpose: App Runner has no arm64 runtime.
#
# Two things happen at build time so that a cold container is useful in the first
# second a judge looks at it:
#   1. the YOLOX-tiny ONNX weights are fetched and their digest printed, so the
#      container never downloads a model at request time;
#   2. `keepout demo --bake` runs the real pipeline over EVERY bundled clip and
#      writes the results plus their evidence frames into the image, so `/api/demo`
#      is a file read rather than a wait. This matters more than it sounds: App
#      Runner runs YOLOX-tiny at about 390 ms a frame, so analysing a 24 second
#      clip live on the instance takes over a minute.

FROM --platform=linux/amd64 python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENCV26_MODEL_DIR=/opt/models \
    KEEPOUT_SAMPLES_DIR=/app/products/keepout/data/samples \
    KEEPOUT_DEMO_DIR=/app/products/keepout/data/demo \
    OPENCV26_UPLOAD_DIR=/tmp/keepout-uploads

# opencv-python-headless still needs libGL-free glib and the video codecs its
# bundled FFmpeg links against. `headless` is what keeps the image small: the
# GUI build pulls in the whole X stack for an imshow we never call.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      ca-certificates \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the whole tree.
COPY products/keepout/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# The shared workspace packages, then the product.
COPY packages/visioncore /app/packages/visioncore
COPY packages/servicekit /app/packages/servicekit
RUN python -m pip install --no-cache-dir --no-deps \
      /app/packages/visioncore /app/packages/servicekit

COPY products/keepout /app/products/keepout
RUN python -m pip install --no-cache-dir --no-deps /app/products/keepout

# Fail loudly and early if the wheel resolved to OpenCV 4.x. The competition
# requires OpenCV 5 and an unpinned install can quietly land on 4.14.
RUN python -c "import cv2, sys; \
    assert cv2.__version__.startswith('5.'), f'expected OpenCV 5, got {cv2.__version__}'; \
    print('OpenCV', cv2.__version__)"

# Fetch and verify the detector at build time. Apache-2.0, Megvii YOLOX-tiny.
RUN mkdir -p /opt/models \
 && python -c "from visioncore import fetch_model, YOLOX_TINY; \
    p = fetch_model(YOLOX_TINY); print('model', p, p.stat().st_size, 'bytes')"

# Bake the demo runs: real analysis, executed here, served instantly at runtime.
# `--clip all` writes data/demo/<clip>/run.json plus data/demo/index.json.
RUN keepout demo --bake --clip all \
 && cat /app/products/keepout/data/demo/index.json \
 && ls -1 /app/products/keepout/data/demo/*/evidence/*.jpg | wc -l

RUN keepout version

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# One worker. The pipeline is CPU-bound and servicekit already runs the analyzer
# off the event loop in a thread, so a second worker would only contend for the
# same two vCPUs and double the resident model.
CMD ["uvicorn", "keepout.service:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75"]

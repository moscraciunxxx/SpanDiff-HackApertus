FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/spandiff
COPY track_2a/requirements.txt /tmp/requirements.txt
# Keep torch on --index-url. An extra index still prefers the PyPI wheel, which is not the CPU build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"

COPY . /opt/spandiff
WORKDIR /opt/spandiff
# CPU torch is installed because the official evaluator imports torch.
# This command grades track_2a/predictions/dev with --split dev.
# It does not load model weights. If the checkout is absent at runtime,
# docker_grade.sh fetches the development gold and the grader over HTTPS.
CMD ["bash", "track_2a/docker_grade.sh"]

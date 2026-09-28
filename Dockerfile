FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
# Demo settings for small free hosts: sample data, tiny face model, smaller detector
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH DEMO=1 MODEL=buffalo_sc DET=320
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# bake the face model into the image so startup does not re-download it
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_sc', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(320,320))"
COPY --chown=user app.py dashboard.html ./
# hosts like Render inject $PORT; default 7860 elsewhere
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

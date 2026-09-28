# Face Attendance System

Live face-recognition attendance with student registration, schedules that open and close attendance automatically,
filterable reports and Excel export. FastAPI + InsightFace (ArcFace) + a single-page dashboard.

## Run locally
    pip install -r requirements.txt
    uvicorn app:app --port 8000        # open http://localhost:8000

Settings (environment variables): `MODEL` (buffalo_l default, buffalo_sc for tiny hosts), `DET` (detector size, default 640),
`DEMO=1` (sample data, 20-student cap, resets on restart).

## Deploy (Docker)
The Dockerfile runs in demo mode with the small model. Works on any Docker host that gives at least 512 MB RAM.

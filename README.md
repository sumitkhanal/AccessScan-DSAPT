# AccessScan
## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure your trained model is in place
```
runs/detect/accessscan-v1/weights/best.pt
```
(This is the output of Sprint 2 training. Set the `MODEL_PATH` env var if it's elsewhere.)

### 3. Start the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the web UI
Open `static/index.html` in your browser, OR visit:
```
http://localhost:8000/static/index.html
```

### 5. Run an audit
- Upload an image of a bus stop, tram stop, or train platform
- Optionally enter the site name
- Click **Run Accessibility Audit**
- A PDF report will be generated and downloaded automatically

## API Usage (curl)

```bash
curl -X POST http://localhost:8000/audit \
  -F "file=@/path/to/your/station_photo.jpg" \
  -F "site_name=Flinders Street Station" \
  --output report.pdf
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /audit | Upload image → PDF report |
| GET | /audit/{job_id} | Re-download previous report |
| GET | /health | Check if model is loaded |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | `runs/detect/accessscan-v1/weights/best.pt` | Path to trained YOLOv8 weights |

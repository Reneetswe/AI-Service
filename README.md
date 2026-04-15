# AI Service

This is a standalone AI service that can be integrated with the React frontend.

## Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your model files**:
   - Place your model files (e.g., `best.pt`, `data.yaml`) in the `models/` directory

4. **Run the service**:
   ```bash
   python main.py
   ```

5. **Access the API**:
   - Base URL: `http://localhost:8000`
   - Test endpoint: `GET /`
   - Prediction endpoint: `POST /predict`

## Development

- The server will automatically reload when you make changes to the code
- For production, consider using a production ASGI server like Gunicorn with Uvicorn workers

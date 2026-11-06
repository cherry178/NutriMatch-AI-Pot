#!/bin/bash
echo "=============================================================================="
echo "Adaptive Personalized Recipe Recommendation Engine"
echo "=============================================================================="
echo "Team Members:"
echo "  - Krrish Jha (24STUCHH010116)"
echo "  - Rasineni Charishma (24STUCHH010662)"
echo "  - Ande Hiranmai (24STUCHH010931)"
echo "  - Chirra Chaithanya (24STUCHH010120)"
echo "=============================================================================="

echo "Installing requirements..."
python3 -m pip install -r requirements.txt --quiet

echo ""
echo "Starting FastAPI & Web Dashboard on http://127.0.0.1:8000 ..."
echo "Open http://127.0.0.1:8000 in your browser."
echo "Press Ctrl+C to stop the server."
echo ""

python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

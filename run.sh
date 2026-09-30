#!/usr/bin/env bash
# run.sh — Start Varsh-AI backend + frontend together
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

echo "╔══════════════════════════════════════════════╗"
echo "║        VARSH-AI — Starting Services           ║"
echo "╚══════════════════════════════════════════════╝"

# 1. Ingest real IMD NetCDF data & train models if not already done
if [ ! -f "$BACKEND/data/district_data.json" ]; then
  echo ""
  echo "🔧 Ingesting IMD 0.25° NetCDF files & training models (first run)..."
  cd "$BACKEND"
  python3 data/ingest_imd.py
fi

if [ ! -f "$BACKEND/data/districts_all_data.json" ]; then
  echo ""
  echo "🗺️ Computing 640+ district zonal raster statistics & GeoJSON..."
  cd "$BACKEND"
  python3 data/zonal_statistics.py
fi


# Clean up any lingering processes on ports 8000 or 3000
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true

# 2. Start FastAPI backend
echo ""
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
cd "$BACKEND"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 3

# Check backend started
if curl -sf http://localhost:8000/ > /dev/null; then
  echo "   ✅ Backend online (PID $BACKEND_PID)"
else
  echo "   ❌ Backend failed to start"
  exit 1
fi

# 3. Start Next.js frontend
echo ""
echo "🌐 Starting Next.js frontend on http://localhost:3000 ..."
cd "$FRONTEND"
npm run dev -- -p 3000 &
FRONTEND_PID=$!


echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Varsh-AI is running!                        ║"
echo "║  Frontend : http://localhost:3000            ║"
echo "║  API      : http://localhost:8000            ║"
echo "║  API Docs : http://localhost:8000/docs       ║"
echo "║                                              ║"
echo "║  Ctrl+C to stop all services                ║"
echo "╚══════════════════════════════════════════════╝"

# Wait and clean up on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait

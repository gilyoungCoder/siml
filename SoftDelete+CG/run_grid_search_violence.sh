#!/bin/bash
# ============================================================================
# Violence Grid Search - Nohup Wrapper
# 원격 접속이 끊겨도 계속 실행되도록 nohup 사용
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/grid_search"
LOG_FILE="${LOG_DIR}/violence_${TIMESTAMP}.log"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "============================================"
echo "🚀 Starting Violence Grid Search (Background)"
echo "============================================"
echo "Log file: $LOG_FILE"
echo "PID file: ${LOG_DIR}/violence.pid"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if running:"
echo "  cat ${LOG_DIR}/violence.pid"
echo "  ps -p \$(cat ${LOG_DIR}/violence.pid 2>/dev/null) 2>/dev/null"
echo ""
echo "To stop:"
echo "  kill \$(cat ${LOG_DIR}/violence.pid)"
echo "============================================"

# nohup으로 백그라운드 실행
nohup bash grid_search_violence.sh > "$LOG_FILE" 2>&1 &

# PID 저장
PID=$!
echo $PID > "${LOG_DIR}/violence.pid"

echo ""
echo "✅ Started with PID: $PID"
echo "   You can safely disconnect now!"
echo ""
echo "Waiting 5 seconds to check if started successfully..."
sleep 5

if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process is running!"
    echo ""
    echo "First few lines of log:"
    head -20 "$LOG_FILE"
else
    echo "❌ Process failed to start. Check log:"
    cat "$LOG_FILE"
    exit 1
fi

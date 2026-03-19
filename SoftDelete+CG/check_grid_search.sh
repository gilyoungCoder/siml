#!/bin/bash
# ============================================================================
# Grid Search 상태 확인 스크립트
# ============================================================================

LOG_DIR="logs/grid_search"

echo "============================================"
echo "📊 Grid Search Status Check"
echo "============================================"
echo ""

# Nudity 상태 확인
echo "🔹 Nudity Grid Search:"
if [ -f "${LOG_DIR}/nudity.pid" ]; then
    PID=$(cat "${LOG_DIR}/nudity.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "   ✅ RUNNING (PID: $PID)"

        # 최신 로그 파일 찾기
        LATEST_LOG=$(ls -t ${LOG_DIR}/nudity_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "   📝 Log: $LATEST_LOG"

            # 진행률 확인
            TOTAL=$(grep "Total experiments:" "$LATEST_LOG" | tail -1 | awk '{print $NF}')
            COMPLETED=$(grep -c "✅ Experiment.*complete" "$LATEST_LOG")

            if [ -n "$TOTAL" ] && [ -n "$COMPLETED" ]; then
                PERCENT=$((COMPLETED * 100 / TOTAL))
                echo "   📈 Progress: $COMPLETED / $TOTAL ($PERCENT%)"
            fi

            # 마지막 업데이트
            LAST_LINE=$(tail -1 "$LATEST_LOG")
            echo "   🕐 Last update: $(stat -c %y "$LATEST_LOG" | cut -d. -f1)"

            # 가장 최근 실험
            LAST_EXP=$(grep "Experiment \[" "$LATEST_LOG" | tail -1)
            if [ -n "$LAST_EXP" ]; then
                echo "   🔄 Current: $LAST_EXP"
            fi
        fi
    else
        echo "   ❌ STOPPED (last PID: $PID)"
        echo "   Check log: $(ls -t ${LOG_DIR}/nudity_*.log 2>/dev/null | head -1)"
    fi
else
    echo "   ⚪ NOT STARTED"
fi

echo ""

# Violence 상태 확인
echo "🔹 Violence Grid Search:"
if [ -f "${LOG_DIR}/violence.pid" ]; then
    PID=$(cat "${LOG_DIR}/violence.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "   ✅ RUNNING (PID: $PID)"

        # 최신 로그 파일 찾기
        LATEST_LOG=$(ls -t ${LOG_DIR}/violence_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "   📝 Log: $LATEST_LOG"

            # 진행률 확인
            TOTAL=$(grep "Total experiments:" "$LATEST_LOG" | tail -1 | awk '{print $NF}')
            COMPLETED=$(grep -c "✅ Experiment.*complete" "$LATEST_LOG")

            if [ -n "$TOTAL" ] && [ -n "$COMPLETED" ]; then
                PERCENT=$((COMPLETED * 100 / TOTAL))
                echo "   📈 Progress: $COMPLETED / $TOTAL ($PERCENT%)"
            fi

            # 마지막 업데이트
            echo "   🕐 Last update: $(stat -c %y "$LATEST_LOG" | cut -d. -f1)"

            # 가장 최근 실험
            LAST_EXP=$(grep "Experiment \[" "$LATEST_LOG" | tail -1)
            if [ -n "$LAST_EXP" ]; then
                echo "   🔄 Current: $LAST_EXP"
            fi
        fi
    else
        echo "   ❌ STOPPED (last PID: $PID)"
        echo "   Check log: $(ls -t ${LOG_DIR}/violence_*.log 2>/dev/null | head -1)"
    fi
else
    echo "   ⚪ NOT STARTED"
fi

echo ""
echo "============================================"
echo "📁 Output Directories:"
echo "============================================"

# 생성된 실험 폴더 개수
if [ -d "scg_outputs/grid_search_nudity" ]; then
    NUDITY_COUNT=$(ls -1d scg_outputs/grid_search_nudity/gs* 2>/dev/null | wc -l)
    echo "   Nudity experiments: $NUDITY_COUNT folders"
fi

if [ -d "scg_outputs/grid_search_violence" ]; then
    VIOLENCE_COUNT=$(ls -1d scg_outputs/grid_search_violence/gs* 2>/dev/null | wc -l)
    echo "   Violence experiments: $VIOLENCE_COUNT folders"
fi

echo ""
echo "============================================"
echo "💾 Disk Usage:"
echo "============================================"

if [ -d "scg_outputs/grid_search_nudity" ]; then
    NUDITY_SIZE=$(du -sh scg_outputs/grid_search_nudity 2>/dev/null | awk '{print $1}')
    echo "   Nudity: $NUDITY_SIZE"
fi

if [ -d "scg_outputs/grid_search_violence" ]; then
    VIOLENCE_SIZE=$(du -sh scg_outputs/grid_search_violence 2>/dev/null | awk '{print $1}')
    echo "   Violence: $VIOLENCE_SIZE"
fi

echo ""
echo "============================================"
echo "💡 Useful Commands:"
echo "============================================"
echo "   Monitor nudity:   tail -f \$(ls -t ${LOG_DIR}/nudity_*.log 2>/dev/null | head -1)"
echo "   Monitor violence: tail -f \$(ls -t ${LOG_DIR}/violence_*.log 2>/dev/null | head -1)"
echo "   Stop nudity:      kill \$(cat ${LOG_DIR}/nudity.pid)"
echo "   Stop violence:    kill \$(cat ${LOG_DIR}/violence.pid)"
echo "============================================"

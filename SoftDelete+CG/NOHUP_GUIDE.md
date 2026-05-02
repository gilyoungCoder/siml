# Nohup으로 Grid Search 돌리기 가이드

원격 서버 접속이 끊겨도 Grid Search가 계속 실행되도록 설정하는 방법입니다.

---

## 🚀 빠른 시작

### 1. Nudity Grid Search 시작 (백그라운드)
```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

./run_grid_search_nudity.sh
```

**출력 예시:**
```
============================================
🚀 Starting Nudity Grid Search (Background)
============================================
Log file: logs/grid_search/nudity_20241224_143022.log
PID file: logs/grid_search/nudity.pid

✅ Started with PID: 12345
   You can safely disconnect now!
```

### 2. Violence Grid Search 시작 (백그라운드)
```bash
./run_grid_search_violence.sh
```

### 3. 둘 다 동시에 시작 (병렬 실행)
```bash
./run_grid_search_nudity.sh
./run_grid_search_violence.sh
```

**이제 SSH 접속을 끊어도 됩니다!** 🎉

---

## 📊 진행 상황 확인

### 빠른 상태 확인
```bash
./check_grid_search.sh
```

**출력 예시:**
```
============================================
📊 Grid Search Status Check
============================================

🔹 Nudity Grid Search:
   ✅ RUNNING (PID: 12345)
   📝 Log: logs/grid_search/nudity_20241224_143022.log
   📈 Progress: 45 / 576 (7%)
   🕐 Last update: 2024-12-24 14:42:15
   🔄 Current: Experiment [45/576]: gs8.0_hs1.25_st0.3_ws4.0-1.0_ts0.0--2.0

🔹 Violence Grid Search:
   ✅ RUNNING (PID: 12346)
   📈 Progress: 32 / 576 (5%)
```

### 실시간 로그 모니터링
```bash
# Nudity 로그 실시간 보기
tail -f logs/grid_search/nudity_*.log

# Violence 로그 실시간 보기
tail -f logs/grid_search/violence_*.log

# 종료: Ctrl+C
```

---

## 🛑 중지하기

### Nudity Grid Search 중지
```bash
kill $(cat logs/grid_search/nudity.pid)
```

### Violence Grid Search 중지
```bash
kill $(cat logs/grid_search/violence.pid)
```

### 둘 다 중지
```bash
kill $(cat logs/grid_search/nudity.pid)
kill $(cat logs/grid_search/violence.pid)
```

---

## 🔄 재시작하기

### 완전히 새로 시작
```bash
# 기존 프로세스 중지
kill $(cat logs/grid_search/nudity.pid) 2>/dev/null

# 새로 시작
./run_grid_search_nudity.sh
```

### 중단된 곳부터 이어서 (수동)
Grid search 스크립트는 이미 생성된 폴더를 건너뛰도록 수정할 수 있습니다:

```bash
# grid_search_nudity.sh 수정 (Line 117 이후)
# 실험 실행 전에 체크 추가:

# 폴더가 이미 존재하고 이미지가 있으면 건너뛰기
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A $OUTPUT_DIR/*.png 2>/dev/null)" ]; then
    echo "⏭️  Skipping (already exists): $EXPERIMENT_NAME"
    continue
fi

# 실험 실행
python generate_selective_cg.py ...
```

---

## 📁 로그 파일 관리

### 로그 파일 위치
```
logs/grid_search/
├── nudity_20241224_143022.log    # 전체 로그
├── nudity.pid                     # 프로세스 ID
├── violence_20241224_143025.log
└── violence.pid
```

### 로그 파일 보기
```bash
# 최신 nudity 로그
cat logs/grid_search/nudity_*.log | less

# 최근 50줄만
tail -50 logs/grid_search/nudity_*.log

# 에러만 검색
grep -i error logs/grid_search/nudity_*.log
```

### 오래된 로그 정리
```bash
# 7일 이상 된 로그 삭제
find logs/grid_search -name "*.log" -mtime +7 -delete
```

---

## 💡 유용한 팁

### 1. 화면에서 로그 보면서 실행
```bash
# tmux 사용 (권장!)
tmux new -s grid_search_nudity
./run_grid_search_nudity.sh

# tmux 세션에서 나가기: Ctrl+B, D
# 다시 들어가기: tmux attach -t grid_search_nudity
```

### 2. 여러 터미널 동시에 모니터링
```bash
# Terminal 1: Nudity 실시간 로그
tail -f logs/grid_search/nudity_*.log

# Terminal 2: Violence 실시간 로그
tail -f logs/grid_search/violence_*.log

# Terminal 3: 상태 확인 (5초마다)
watch -n 5 ./check_grid_search.sh
```

### 3. 디스크 공간 모니터링
```bash
# 5분마다 디스크 사용량 체크
watch -n 300 'df -h | grep -E "Filesystem|/mnt"'
```

### 4. 알림 받기 (완료 시 이메일)
```bash
# grid_search_nudity.sh 마지막에 추가
echo "Nudity grid search completed!" | mail -s "Grid Search Done" your@email.com
```

---

## 🔍 문제 해결

### 프로세스가 자꾸 멈춤
```bash
# GPU 메모리 확인
watch -n 1 nvidia-smi

# OOM 발생 시 NSAMPLES 줄이기
# grid_search_nudity.sh Line 21
NSAMPLES=1
```

### 로그가 너무 큼
```bash
# Visualization 끄기 (용량 절약)
# grid_search_nudity.sh Line 150
# --save_visualizations 제거
```

### SSH 접속 끊김 방지 (예방)
SSH 클라이언트 설정에서 keep-alive 설정:

**~/.ssh/config** (로컬 컴퓨터)
```
Host your-server
    ServerAliveInterval 60
    ServerAliveCountMax 30
```

---

## 📊 완료 확인

### Grid Search 완료 여부 확인
```bash
# Nudity 완료 확인 (576 experiments)
ls -1d scg_outputs/grid_search_nudity/gs* | wc -l

# Violence 완료 확인 (576 experiments)
ls -1d scg_outputs/grid_search_violence/gs* | wc -l

# 로그에서 완료 메시지 확인
tail -20 logs/grid_search/nudity_*.log
```

**완료 메시지:**
```
🎉 Grid Search Complete!
Total experiments: 576
```

### 결과 분석
```bash
# 완료 후 분석
python analyze_grid_search.py scg_outputs/grid_search_nudity/ \
    --output nudity_results.csv

python analyze_grid_search.py scg_outputs/grid_search_violence/ \
    --output violence_results.csv
```

---

## 🎯 권장 워크플로우

### 1단계: 시작
```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# 둘 다 시작
./run_grid_search_nudity.sh
./run_grid_search_violence.sh

# 상태 확인
./check_grid_search.sh
```

### 2단계: 접속 끊기
```
접속 종료해도 OK!
다음날 다시 접속해서 확인
```

### 3단계: 재접속 후 확인
```bash
ssh your-server
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# 상태 확인
./check_grid_search.sh

# 로그 확인
tail -50 logs/grid_search/nudity_*.log
```

### 4단계: 완료 후 분석
```bash
# 결과 분석
python analyze_grid_search.py scg_outputs/grid_search_nudity/
python analyze_grid_search.py scg_outputs/grid_search_violence/

# 최적 파라미터 찾기
# CSV 파일 확인
```

---

## 📞 자주 묻는 질문

**Q: 실행 중인지 어떻게 확인하나요?**
```bash
./check_grid_search.sh
```

**Q: 로그를 실시간으로 보고 싶어요.**
```bash
tail -f logs/grid_search/nudity_*.log
```

**Q: 중간에 멈춘 것 같아요.**
```bash
# 프로세스 확인
ps -p $(cat logs/grid_search/nudity.pid)

# 로그 마지막 확인
tail -20 logs/grid_search/nudity_*.log

# 필요시 재시작
kill $(cat logs/grid_search/nudity.pid)
./run_grid_search_nudity.sh
```

**Q: 예상 완료 시간은?**
```
576 experiments × 2분/exp = 19.2시간 (약 하루)
```

**Q: 디스크 공간이 부족해요.**
```bash
# Visualization 제거
find scg_outputs/grid_search_* -name "visualizations" -type d -exec rm -rf {} +

# 또는 grid_search_*.sh에서 --save_visualizations 제거
```

---

## ✅ 체크리스트

실행 전:
- [ ] GPU 확인: `nvidia-smi`
- [ ] 디스크 공간: `df -h` (최소 60GB 필요)
- [ ] Classifier 경로 확인
- [ ] Prompt 파일 확인

실행:
- [ ] `./run_grid_search_nudity.sh`
- [ ] `./run_grid_search_violence.sh`
- [ ] 상태 확인: `./check_grid_search.sh`
- [ ] 로그 확인: `tail -f logs/grid_search/nudity_*.log`

주기적 확인 (하루 1-2회):
- [ ] 상태 확인: `./check_grid_search.sh`
- [ ] 디스크 사용량 확인
- [ ] GPU 사용률 확인

완료 후:
- [ ] 결과 분석: `python analyze_grid_search.py ...`
- [ ] 최적 파라미터 선정
- [ ] `run_full_adaptive.sh` 업데이트

---

**이제 안심하고 접속을 끊으셔도 됩니다!** 🎉

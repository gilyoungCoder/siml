# GenEval Quick Installation Guide

GenEval이 `mmdet` 모듈을 찾지 못하는 문제를 해결하는 방법입니다.

## 문제

```
ModuleNotFoundError: No module named 'mmdet'
```

## 해결 방법

현재 `sdd` conda 환경을 사용 중이므로, 이 환경에 GenEval 의존성을 설치하면 됩니다.

### 방법 1: pip로 간단 설치 (권장)

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# MMDetection 3.x 설치 (최신 버전)
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 기타 의존성
pip install open-clip-torch clip-benchmark networkx

# 모델 다운로드
cd geneval
./evaluation/download_models.sh models/
cd ..
```

### 방법 2: 전체 자동 설치

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
bash install_geneval.sh
```

### 방법 3: GenEval 환경 파일 사용

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG/geneval
conda env update -f environment.yml -n sdd
```

## 설치 확인

설치 후 확인:

```bash
python -c "import mmdet; print('mmdet version:', mmdet.__version__)"
python -c "import mmcv; print('mmcv version:', mmcv.__version__)"
```

성공하면:
```
mmdet version: 3.x.x
mmcv version: 2.x.x
```

## 모델 다운로드

Object detector 모델 다운로드:

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG/geneval
mkdir -p models
./evaluation/download_models.sh models/
```

다운로드된 파일 확인:
```bash
ls -lh geneval/models/
```

## 실행

설치 완료 후:

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
bash run_geneval.sh
```

## 문제 해결

### 1. CUDA 버전 불일치

```bash
# 현재 CUDA 버전 확인
nvcc --version

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"
```

CUDA 11.8을 사용 중이면:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1을 사용 중이면:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. mmcv 설치 실패

구버전 시도:
```bash
mim install mmcv-full==1.7.2
```

### 3. Python 버전 문제

GenEval은 Python 3.8-3.10을 권장합니다. 현재 Python 3.13을 사용 중이라면, 새 환경 생성:

```bash
conda create -n geneval python=3.10
conda activate geneval
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
bash install_geneval.sh
```

## 최소 설치 (빠른 테스트용)

최소한의 패키지만 설치:

```bash
pip install mmengine mmcv mmdet open-clip-torch
cd geneval && ./evaluation/download_models.sh models/ && cd ..
bash run_geneval.sh
```

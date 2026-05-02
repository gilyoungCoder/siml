# geo_utils/sae_probe.py
import os
import sys
import csv
import torch

# SAeUron repo가 현재 프로젝트 밖에 있으니, 경로 보정 (필요시 수정)
# 기본 위치 예: ~/unlearning/SAeUron
_DEFAULT_SAEURON_DIR = os.path.expanduser("~/unlearning/SAeUron")
try:
    from SAE.sae import Sae  # SAeUron/SAE/sae.py
except ModuleNotFoundError:
    if os.path.isdir(_DEFAULT_SAEURON_DIR):
        sys.path.append(_DEFAULT_SAEURON_DIR)
        from SAE.sae import Sae
    else:
        raise


class SAEProbe:
    """
    - Diffusers UNet 내부 모듈(hookpoint)에 forward hook을 달아 feature map을 잡고
    - SAeUron SAE로 encoding z를 구한 뒤
    - 선택된 feature 집합 Fc의 평균 활성(FAI)을 스텝별로 기록
    """

    def __init__(
        self,
        pipe,
        sae_repo: str = "bcywinski/SAeUron_nudity",
        hookpoint: str = "up_blocks.1.attentions.1",  # pipe.unet 기준 경로
        device: str = "cuda",
        topk_select: int = 32,
        csv_path: str | None = None,
    ):
        self.pipe = pipe
        self.device = device
        self.hookpoint = hookpoint
        # 허깅페이스 허브에서 SAE 로드
        self.sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint, device=device)

        self.buffer = None  # 훅에서 받은 텐서
        self.z_log = []     # (prompt_idx, step, timestep, FAI)
        self.topk_select = topk_select
        self.fc = None      # 선택된 latent feature index list
        self.csv_path = csv_path

        self._register_hook()

    # -------- internal utils --------
    @staticmethod
    def _pick_tensor_from_output(out):
        """diffusers 모듈의 출력(out)이 Tensor가 아닐 수 있으므로 안전하게 Tensor만 추출."""
        # ModelOutput 형태: .sample 우선
        if hasattr(out, "sample") and isinstance(out.sample, torch.Tensor):
            return out.sample
        # dict: 'sample' 우선, 없으면 첫 텐서
        if isinstance(out, dict):
            if "sample" in out and isinstance(out["sample"], torch.Tensor):
                return out["sample"]
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            return None
        # tuple/list: 첫 텐서
        if isinstance(out, (tuple, list)):
            for v in out:
                if isinstance(v, torch.Tensor):
                    return v
            return None
        # 텐서 그대로
        if isinstance(out, torch.Tensor):
            return out
        return None

    def _register_hook(self):
        # pipe.unet 기준 상대 경로로 변환 (허브에는 'unet.' 포함해 저장된 경우도 있으므로 제거)
        path = self.hookpoint.strip()
        if path.startswith("unet."):
            path = path[len("unet."):]

        # 1차 시도: 그대로
        try:
            mod = self.pipe.unet.get_submodule(path)
        except Exception:
            # 2차 시도: transformer_blocks.0 까지 들어가야 하는 경우
            try:
                mod = self.pipe.unet.get_submodule(path + ".transformer_blocks.0")
            except Exception as e:
                raise AttributeError(
                    f"[SAEProbe] Can't find hookpoint '{self.hookpoint}'. "
                    f"Use paths relative to 'pipe.unet', e.g. "
                    f"'up_blocks.1.attentions.1' or 'up_blocks.1.attentions.2'. "
                    f"Original error: {e}"
                )

        def _hook(_module, _inp, out):
            self.buffer = self._pick_tensor_from_output(out)
        self.hook_handle = mod.register_forward_hook(_hook)

    def _ensure_tensor(self, feat):
        """hook buffer를 다시 한 번 안전하게 Tensor로 변환 (이중 방어)."""
        if feat is None:
            return None
        t = self._pick_tensor_from_output(feat)
        if isinstance(t, torch.Tensor):
            return t
        # 이미 텐서였던 경우
        if isinstance(feat, torch.Tensor):
            return feat
        return None

    def _encode_to_z(self, feat):
        """
        feat: (B,C,H,W) 또는 (B,S,C)를 받아 SAE.encode에 (B,S,C)로 넣음.
        반환 z가 (B,S,F)이면 S-mean -> (B,F)로.
        """
        feat = self._ensure_tensor(feat)
        if feat is None:
            return None

        feat = feat.to(self.device)

        # 1) SAE 입력을 (B, S, C)로 맞춤
        if feat.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = feat.shape
            x_seq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        elif feat.dim() == 3:
            # 이미 (B, S, C)
            x_seq = feat.contiguous()
            B, S, C = x_seq.shape
            H = W = None
        else:
            return None

        # 2) SAE 인코딩 (SAeUron은 (B, S, C) 입력을 기대)
        with torch.no_grad():
            out = None
            if hasattr(self.sae, "encode"):
                out = self.sae.encode(x_seq)
            elif hasattr(self.sae, "forward_enc"):
                out = self.sae.forward_enc(x_seq)
            else:
                out = self.sae(x_seq)

            # out이 tuple/list인 경우 첫번째를 z로 사용
            if isinstance(out, (tuple, list)):
                z = out[0]
            else:
                z = out

        # 3) 반환 텐서 정규화: (B,S,F)->(B,F), (B,F)->(B,F)
        if not isinstance(z, torch.Tensor):
            return None

        if z.dim() == 3:
            # (B, S, F) -> (B, F)
            z = z.mean(dim=1)
        elif z.dim() == 2:
            # 이미 (B, F)
            pass
        else:
            # 예외 형태 방어 (필요 시 추가 처리)
            return None

        return z


    # -------- public APIs --------
    @torch.no_grad()
    def calibrate_fc(self, nudity_prompts, neutral_prompts, pipe_kwargs_base):
        """
        간단 캘리브레이션:
        score(i) = mean_z(i|nudity) - mean_z(i|neutral)
        상위 topk_select 인덱스를 Fc로 선택
        """
        def _run(prompts):
            all_z = []
            for p in prompts:
                kws = dict(pipe_kwargs_base)
                kws.update(dict(prompt=p))
                self.buffer = None
                _ = self.pipe(**kws)
                z = self._encode_to_z(self.buffer)
                if z is None:
                    continue
                all_z.append(z)
            if not all_z:
                return None
            return torch.cat(all_z, dim=0).mean(dim=0)  # (n_latent,)

        z_nud = _run(nudity_prompts)
        z_neu = _run(neutral_prompts)
        if z_nud is None or z_neu is None:
            print("[SAEProbe] calibration failed (no SAE buffer).")
            return
        score = (z_nud - z_neu)
        k = min(self.topk_select, score.numel())
        self.fc = torch.topk(score, k=k, largest=True).indices.tolist()
        print(f"[SAEProbe] selected {len(self.fc)} features for nudity.")

    def log_step(self, prompt_idx, step, timestep):
        if self.buffer is None:
            return
        z = self._encode_to_z(self.buffer)
        if z is None:
            return
        z_mean = z.mean(dim=0)  # (n_latent,)
        if self.fc is None:
            fai = float(z_mean.mean().item())
        else:
            fai = float(z_mean[self.fc].mean().item())
        self.z_log.append((prompt_idx, int(step), int(timestep), fai))

    def flush_csv(self):
        if not self.csv_path or not self.z_log:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_idx", "step", "timestep", "FAI"])
            w.writerows(self.z_log)

    def close(self):
        if hasattr(self, "hook_handle"):
            self.hook_handle.remove()

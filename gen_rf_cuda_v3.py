# gen_rf_cuda_v3.py
"""
P.mat 설정과 일관된 RF 데이터 생성 (CUDA 가속 버전)
- P.mat에서 모든 파라미터 로드
- sources.mat에서 source 위치/타입 로드
- Band-limited RF wavelet 적용 및 CUDA 기반 병렬 연산 최적화
- MATLAB 호환성을 위한 Time-major 저장 레이아웃 적용
"""
import math
import os
import numpy as np
import torch
import scipy.io as sio
from scipy.io import loadmat

PASSIVE_MODE = True

# =============================================================================
# Configuration
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "Data_tus", "01_sim")
P_MAT = os.path.join(DATA_DIR, "P.mat")
SOURCES_MAT = os.path.join(DATA_DIR, "sources.mat")
OUT_DIR = os.path.join(DATA_DIR, "RfData")
N_BURST = 30

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Helpers
# =============================================================================
def mat_struct_to_dict(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == np.object_:
        if obj.size == 1:
            return mat_struct_to_dict(obj.item())
        return [mat_struct_to_dict(x) for x in obj.flat]
    if hasattr(obj, "_fieldnames"):
        d = {}
        for fn in obj._fieldnames:
            d[fn] = mat_struct_to_dict(getattr(obj, fn))
        return d
    return obj

def load_P(path):
    md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat_struct_to_dict(md["P"])

def load_sources(path):
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    def _to_1d_float32(v):
        return np.asarray(v, dtype=np.float32).reshape(-1)
    def _to_1d_uint8(v):
        return np.asarray(v, dtype=np.uint8).reshape(-1)
    return {
        "src_x_mm": _to_1d_float32(md["src_x_mm"]),
        "src_z_mm": _to_1d_float32(md["src_z_mm"]),
        "src_amp":  _to_1d_float32(md["src_amp"]),
        "src_type": _to_1d_uint8(md["src_type"]),
    }

def make_stable_kernel(fs, f0, dur_us=15.0):
    dur = dur_us * 1e-6
    n = max(int(np.round(dur * fs)), 64)
    t = np.arange(n, dtype=np.float32) / fs
    win = np.hanning(n).astype(np.float32)
    s = np.sin(2 * np.pi * (f0 / 2) * t) + 0.6 * np.sin(2 * np.pi * f0 * t)
    k = (s * win).astype(np.float32)
    k = k / (np.sqrt((k * k).sum()) + 1e-12)
    return torch.from_numpy(k).to(DEVICE)

def fft_conv1d_same(x, k):
    """FFT-based 1D convolution, same-length output (C, T)"""
    C, T = x.shape
    K = k.numel()
    L = T + K - 1
    X = torch.fft.rfft(x, n=L)
    Kf = torch.fft.rfft(k, n=L)
    Y = X * Kf
    y = torch.fft.irfft(Y, n=L)
    start = (K - 1) // 2
    return y[:, start:start + T]

# =============================================================================
# Main
# =============================================================================
def main():
    print(f"USING DEVICE: {DEVICE}")
    P = load_P(P_MAT)
    
    stRfInfo = P["CAV"]["stRfInfo"]
    FUS = P["FUS"]
    stTrans = P["stTrans"]
    
    c = float(stRfInfo["nSoundSpeed"])
    nSample = int(stRfInfo["nSample"])
    nCh = int(stRfInfo["nChannel"])
    fs = float(stRfInfo["nFs"])
    f0 = float(stRfInfo["nFc"])
    nPulse = int(FUS["nNumPulse"])
    
    # Transducer positions
    ele_x = np.array(stTrans["aElePos"], dtype=np.float32).reshape(-1)
    if np.max(np.abs(ele_x)) > 0.5: ele_x *= 1e-3
    ele_x_t = torch.from_numpy(ele_x).to(DEVICE)
    
    lens_m = float(stTrans.get("nLensCorr_m", 0.0))
    if abs(lens_m) > 0.01: lens_m *= 1e-3

    # Source data
    src = load_sources(SOURCES_MAT)
    src_x_t = torch.from_numpy(src["src_x_mm"] * 1e-3).to(DEVICE)
    src_z_t = torch.from_numpy(src["src_z_mm"] * 1e-3).to(DEVICE)
    src_amp_t = torch.from_numpy(src["src_amp"]).to(DEVICE)
    src_type_t = torch.from_numpy(src["src_type"].astype(np.int64)).to(DEVICE)
    nSrc = src_x_t.size(0)

    # Offset Calculation
    offset_m = 0.0
    for key in ["nOffset_m", "offset_m", "nOffsetM", "nOffset"]:
        if key in stRfInfo: offset_m = float(stRfInfo[key]); break
        elif key in P["CAV"]: offset_m = float(P["CAV"][key]); break

    # 1. Build Band-limited RF Wavelet (GPU)
    burst_cycles = int(os.environ.get("RF_BURST_CYCLES", "5"))
    spc = fs / f0
    wave_len = int(max(8, round(burst_cycles * spc)))
    if wave_len % 2 == 0: wave_len += 1
    wave_center = wave_len // 2

    t_wave = (torch.arange(wave_len, device=DEVICE, dtype=torch.float32) - wave_center) / fs
    wave = torch.sin(2.0 * math.pi * f0 * t_wave)
    win = 0.5 - 0.5 * torch.cos(2.0 * math.pi * torch.arange(wave_len, device=DEVICE) / (wave_len - 1))
    wave = (wave * win)
    wave = wave / (wave.abs().max() + 1e-12)

    # 2. Distance and Delay Pre-calculation (GPU)
    dx = src_x_t[:, None] - ele_x_t[None, :]
    dz = src_z_t[:, None] - torch.zeros_like(ele_x_t)[None, :] # ele_z assumed 0
    rx_dist = torch.sqrt(dx * dx + dz * dz) + 1e-9
    
    tx_delay_t = torch.zeros_like(src_z_t) # Passive assumption
    total_delay_m = tx_delay_t[:, None] + rx_dist + lens_m - offset_m
    tau = total_delay_m / c
    
    n0f = tau * fs
    n0i = torch.floor(n0f).to(torch.int64)
    frac = (n0f - n0i.to(n0f.dtype)).clamp(0.0, 1.0)
    
    valid = (n0i >= wave_center) & (n0i < (nSample - (wave_len - wave_center)))
    att = 1.0 / rx_dist

    # 3. Burst Generation Loop
    k_stable = make_stable_kernel(fs, f0)
    P_ACTIVE_STABLE, P_ACTIVE_INERT = 0.7, 0.5

    for b in range(1, N_BURST + 1):
        rf_all_pulses = []
        for p in range(nPulse):
            torch.manual_seed(b * 1000 + p)

            active_st = (torch.rand(nSrc, device=DEVICE) < P_ACTIVE_STABLE) & (src_type_t == 0)
            active_in = (torch.rand(nSrc, device=DEVICE) < P_ACTIVE_INERT) & (src_type_t == 1)

            rf_st = torch.zeros((nCh, nSample), device=DEVICE, dtype=torch.float32)
            rf_in = torch.zeros((nCh, nSample), device=DEVICE, dtype=torch.float32)

            ch_idx = torch.arange(nCh, device=DEVICE)[None, :].expand(nSrc, nCh)

            for m_type, rf_target in [(active_st, rf_st), (active_in, rf_in)]:
                m = valid & m_type[:, None]
                if not m.any(): continue
                
                a_base = src_amp_t[:, None].expand(nSrc, nCh)[m] * att[m]
                jitter = 0.6 + 0.8 * torch.rand(a_base.shape, device=DEVICE)
                a0 = a_base * jitter * (1.0 - frac[m])
                a1 = a_base * jitter * frac[m]
                
                target_ch = ch_idx[m]
                target_n0 = n0i[m]

                for k in range(wave_len):
                    rf_target.index_put_((target_ch, target_n0 - wave_center + k), 
                                        a0 * wave[k], accumulate=True)
                    rf_target.index_put_((target_ch, target_n0 - wave_center + 1 + k), 
                                        a1 * wave[k], accumulate=True)

            rf_st = fft_conv1d_same(rf_st, k_stable)
            k_in = torch.randn(64, device=DEVICE)
            k_in = k_in / (torch.sqrt((k_in**2).sum()) + 1e-12)
            rf_in = fft_conv1d_same(rf_in, k_in)

            rf_all_pulses.append(rf_st + rf_in)

        rf_all = torch.cat(rf_all_pulses, dim=1)
        rf_all += (0.002 * rf_all.abs().mean()) * torch.randn_like(rf_all)
        
        # -------------------------------------------------------------
        # SAVE LAYOUT (MATLAB compatibility)
        # Many MATLAB pipelines read as time-major: [L, nCh]
        # -------------------------------------------------------------
        scale = (0.8 * 32767.0) / (rf_all.abs().max() + 1e-9)
        out = (rf_all * scale).clamp(-32767, 32767).to(torch.int16)
        out_tm = out.transpose(0, 1).contiguous()   # (L, nCh)
        out_np = out_tm.cpu().numpy()
        
        fname = os.path.join(OUT_DIR, f"RfData_spc_{b:03d}.bin")
        out_np.tofile(fname)
        print(f"  [{b:03d}/{N_BURST}] Saved: {fname} (Time-major layout)")

    print("\nAll RF generation complete on CUDA.")

if __name__ == "__main__":
    main()
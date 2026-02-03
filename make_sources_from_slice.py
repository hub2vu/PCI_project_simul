# make_sources_from_slice.py
import os, numpy as np
from scipy.io import savemat

SLICE_DIR = r"./vessel_sweep_out"
SLICE_IDX = 0          # 예: mask_0000.npy 같은 인덱스
OUT_MAT   = r"./sources.mat"

# (mm 단위로 맞추기) 네 npy는 um 스케일 기반이니까 mm로 변환
OUT_UM = 3.0           # 너 make_sweep_hi3um.py에서 OUT_UM(픽셀 간격)과 일치시켜
um_to_mm = 1e-3

# FUS focus (mm)
x0_mm, z0_mm = 0.0, 6.0
sigma_mm = 0.6

N_SOURCES = 3000       # 현실감 올리려면 수천~수만도 가능(시간/메모리 보고)

mask = np.load(os.path.join(SLICE_DIR, f"mask_{SLICE_IDX:04d}.npy")).astype(bool)
vint = np.load(os.path.join(SLICE_DIR, f"vessel_intensity_{SLICE_IDX:04d}.npy")).astype(np.float32)
# radius_field_um도 필요하면 같이 쓰면 됨
radf = np.load(os.path.join(SLICE_DIR, f"radius_field_um_{SLICE_IDX:04d}.npy")).astype(np.float32)

# 좌표 그리드(픽셀 -> mm)
nz, nx = mask.shape
x_mm = (np.arange(nx) - nx/2.0) * OUT_UM * um_to_mm
z_mm = (np.arange(nz)) * OUT_UM * um_to_mm
XX, ZZ = np.meshgrid(x_mm, z_mm)

# focus 가중치
w_focus = np.exp(-((XX - x0_mm)**2 + (ZZ - z0_mm)**2) / (2*sigma_mm**2)).astype(np.float32)

# 최종 가중치: 혈관+크기+포커스
w = (vint * w_focus) * mask
w_flat = w[mask]
if w_flat.size == 0:
    raise RuntimeError("혈관 mask가 비었음. slice index/경계 확인 필요")

# 확률 분포로 샘플링
p = w_flat / (w_flat.sum() + 1e-12)
idx_all = np.flatnonzero(mask)
pick = np.random.choice(idx_all.size, size=min(N_SOURCES, idx_all.size), replace=False,
                        p=None)  # 먼저 후보 선택
# 후보를 weight 기반으로 다시 고르는 방식(정확도 우선)
# 단순화: weight 기반으로 바로 choice하고 싶으면 idx_all에 대응하는 p를 만들어야 함

# weight 기반 샘플링(정확 버전)
# --- weight -> probability (안정 정규화) ---
idx_all = np.flatnonzero(mask)          # 혈관 픽셀 linear index
w_flat = w[mask].astype(np.float64)     # 혈관 픽셀에 대한 weight

# NaN/Inf 제거 + 음수 제거
w_flat[~np.isfinite(w_flat)] = 0.0
w_flat[w_flat < 0] = 0.0

wsum = w_flat.sum()
M = idx_all.size
K = int(min(N_SOURCES, M))

if M == 0:
    raise RuntimeError("mask가 비었습니다. slice index 확인 필요")
if K <= 0:
    raise RuntimeError("N_SOURCES가 0입니다.")

if wsum <= 0:
    # fallback: weight가 전부 0이면 uniform로 샘플
    pick_lin = np.random.choice(idx_all, size=K, replace=False)
else:
    p = w_flat / wsum
    # 부동소수 오차로 sum이 1에서 약간 벗어날 수 있어서 강제 보정
    p = np.clip(p, 0.0, 1.0)
    p = p / p.sum()

    # np.random.choice는 p 길이가 "선택 집합"과 같아야 하므로 idx_all에 대해서만 선택
    pick_in = np.random.choice(M, size=K, replace=False, p=p)
    pick_lin = idx_all[pick_in]
    
iz = pick_lin // nx
ix = pick_lin % nx

src_x_mm = x_mm[ix].astype(np.float32)
src_z_mm = z_mm[iz].astype(np.float32)

# 소스 진폭: 혈관 반지름 반영(큰 혈관이 더 강함)
src_amp = radf[iz, ix].astype(np.float32)  # um 단위(상대값으로 쓰면 됨)

# stable vs inertial 비율(예: focus 중심은 inertial 비율 높게)
focus_strength = w_focus[iz, ix]
prob_inertial = np.clip((focus_strength - focus_strength.min()) /
                        (focus_strength.max() - focus_strength.min() + 1e-12), 0, 1)
src_type = (np.random.rand(prob_inertial.size) < (0.2 + 0.6*prob_inertial)).astype(np.uint8)
# 1=inertial, 0=stable

savemat(OUT_MAT, {
    "src_x_mm": src_x_mm,
    "src_z_mm": src_z_mm,
    "src_amp":  src_amp,
    "src_type": src_type,
    "x0_mm": np.float32(x0_mm),
    "z0_mm": np.float32(z0_mm),
})
print("saved:", OUT_MAT, "N=", src_x_mm.size)

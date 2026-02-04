# make_sources_from_slice_v2.py
"""
P.mat에서 FUS 설정을 읽어와서 source를 생성하는 통합 스크립트.
- P.mat의 FUS focus 위치, 주파수 등을 그대로 사용
- vessel mask 기반으로 source 분포 생성
- multi-foci 지원
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
import argparse
import matplotlib.pyplot as plt
# =============================================================================
# Configuration
# =============================================================================
# Use __file__ based paths for robustness
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 모든 데이터는 Data_tus/01_sim 경로에서 처리
DATA_DIR = os.path.join(_SCRIPT_DIR, "Data_tus", "01_sim")
P_MAT = os.path.join(DATA_DIR, "P.mat")
SLICE_DIR = os.path.join(_SCRIPT_DIR, "vessel_sweep_out")

OUT_MAT = os.path.join(DATA_DIR, "sources.mat")

# Data 폴더 생성
os.makedirs(DATA_DIR, exist_ok=True)

# vessel mask 픽셀 간격 (um) - make_sweep_hi3um.py와 일치시켜야 함
PIXEL_UM = 3.0

# source 개수
N_SOURCES = 10000

# FUS focus 가중치 sigma (mm) - focus 주변 집중도
SIGMA_MM = 0.6

# stable vs inertial 비율 파라미터
INERTIAL_BASE = 0.2   # focus 밖에서 inertial 비율
INERTIAL_MAX = 0.8    # focus 중심에서 inertial 비율
def pick_point_on_image(img2d, x_mm, z_mm, title):
    """
    img2d: (nz,nx)
    x_mm: (nx,) axis in mm
    z_mm: (nz,) axis in mm
    Returns (x_mm, z_mm) or None (ESC)
    """
    picked = {"pt": None}

    fig, ax = plt.subplots()
    ax.imshow(
        img2d,
        extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],  # keep z increasing downward visually
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    marker = ax.scatter([], [], s=80)

    def on_click(event):
        if event.inaxes != ax:
            return
        picked["pt"] = (float(event.xdata), float(event.ydata))
        marker.set_offsets(np.array([[picked["pt"][0], picked["pt"][1]]]))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key == "escape":
            picked["pt"] = None
            plt.close(fig)

    cid1 = fig.canvas.mpl_connect("button_press_event", on_click)
    cid2 = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    return picked["pt"]


def build_axes_from_stG(stG, nx, nz):
    """
    mask 픽셀 격자(nx,nz)를 PCI grid(stG.aX/aZ) 범위(mm)에 선형 매핑.
    overlay/PCI 좌표 불일치 문제를 줄이기 위해 click 모드 기본으로 사용.
    """
    aX = np.array(stG["aX"]).reshape(-1) * 1000.0  # m -> mm
    aZ = np.array(stG["aZ"]).reshape(-1) * 1000.0  # m -> mm
    x_mm = np.linspace(aX.min(), aX.max(), nx, dtype=np.float32)
    z_mm = np.linspace(aZ.min(), aZ.max(), nz, dtype=np.float32)
    return x_mm, z_mm


# =============================================================================
# Helper: MATLAB struct -> dict
# =============================================================================
def mat_struct_to_dict(obj):
    """Recursively convert scipy.io.loadmat matlab struct objects to dict."""
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
    """Load P.mat and convert to nested dict."""
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat_struct_to_dict(md["P"])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slice-idx", type=int, default=14, help="vessel slice index (mask_####.npy)")
    p.add_argument("--out-mat", type=str, default=OUT_MAT, help="output sources.mat path")
    p.add_argument("--n-sources", type=int, default=N_SOURCES, help="number of sources (sampling mode)")
    p.add_argument("--sigma-mm", type=float, default=SIGMA_MM, help="focus gaussian sigma (mm)")
    p.add_argument("--click", action="store_true", help="click a point on the slice to place FUS source")
    p.add_argument("--single-point", action="store_true",
                   help="if set with --click, save exactly 1 point source at the clicked location")
    p.add_argument("--amp", type=float, default=1.0, help="amplitude for single-point mode (overrides radf)")
    p.add_argument("--src-type", type=int, default=1, choices=[0,1],
                   help="0=stable, 1=inertial (single-point mode)")
    return p.parse_args()

# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    SLICE_IDX = int(args.slice_idx)
    OUT_PATH = args.out_mat
    K_default = int(args.n_sources)
    sigma_mm = float(args.sigma_mm)
    # -------------------------------------------------------------------------
    # 1. P.mat에서 FUS 설정 로드
    # -------------------------------------------------------------------------
    print(f"Loading P.mat from: {P_MAT}")
    P = load_P(P_MAT)
    
    FUS = P["FUS"]
    CAV = P["CAV"]
    stG = CAV["stG"]
    stRfInfo = CAV["stRfInfo"]
    
    # FUS focus 위치 (meters -> mm)
    mFocus_xz_m = np.array(FUS["mFocus_xz_m"])  # (nFoci, 2): [x, z]
    if mFocus_xz_m.ndim == 1:
        mFocus_xz_m = mFocus_xz_m.reshape(1, -1)
    
    foci_x_mm = mFocus_xz_m[:, 0] * 1000  # mm
    foci_z_mm = mFocus_xz_m[:, 1] * 1000  # mm
    n_foci = len(foci_x_mm)
    
    print(f"\n=== FUS Configuration from P.mat ===")
    print(f"Number of foci: {n_foci}")
    for i in range(n_foci):
        print(f"  Foci {i}: x = {foci_x_mm[i]:.2f} mm, z = {foci_z_mm[i]:.2f} mm")
    
    # 중심 focus (multi-foci의 평균 또는 대표값)
    x0_mm = float(foci_x_mm.mean())
    z0_mm = float(foci_z_mm.mean())
    print(f"Center focus: x = {x0_mm:.2f} mm, z = {z0_mm:.2f} mm")
    
    # PCI grid 범위 (확인용)
    aZ = np.array(stG["aZ"])
    aX = np.array(stG["aX"])
    print(f"\nPCI grid range:")
    print(f"  Z: {aZ.min()*1000:.2f} ~ {aZ.max()*1000:.2f} mm")
    print(f"  X: {aX.min()*1000:.2f} ~ {aX.max()*1000:.2f} mm")
    
    # -------------------------------------------------------------------------
    # 2. Vessel mask 로드
    # -------------------------------------------------------------------------
    print(f"\nLoading vessel mask from: {SLICE_DIR}/mask_{SLICE_IDX:04d}.npy")
    
    mask_path = os.path.join(SLICE_DIR, f"mask_{SLICE_IDX:04d}.npy")
    vint_path = os.path.join(SLICE_DIR, f"vessel_intensity_{SLICE_IDX:04d}.npy")
    radf_path = os.path.join(SLICE_DIR, f"radius_field_um_{SLICE_IDX:04d}.npy")
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = np.load(mask_path).astype(bool)
    vint = np.load(vint_path).astype(np.float32) if os.path.exists(vint_path) else np.ones_like(mask, dtype=np.float32)
    radf = np.load(radf_path).astype(np.float32) if os.path.exists(radf_path) else np.ones_like(mask, dtype=np.float32)
    
    nz, nx = mask.shape
    print(f"Mask shape: {mask.shape} (nz={nz}, nx={nx})")
    print(f"Vessel pixels: {mask.sum()}")
    
    # -------------------------------------------------------------------------
    # 3. 좌표 그리드 생성 (pixel -> mm)
    # -------------------------------------------------------------------------
    # CLICK 모드에서는 overlay/PCI 좌표 일치를 위해 stG 범위(mm)로 선형 매핑
    if args.click:
        x_mm, z_mm = build_axes_from_stG(stG, nx=nx, nz=nz)
        print("\nCoordinate mapping (CLICK mode, stG-aligned):")
        print(f"  X range: {x_mm.min():.2f} ~ {x_mm.max():.2f} mm")
        print(f"  Z range: {z_mm.min():.2f} ~ {z_mm.max():.2f} mm")
    else:
        # 기존 방식(픽셀 기반 + focus에 맞춘 z_offset)
        um_to_mm = 1e-3
        x_mm = (np.arange(nx) - nx / 2.0) * PIXEL_UM * um_to_mm
        z_raw_mm = np.arange(nz) * PIXEL_UM * um_to_mm
        z_mask_center = z_raw_mm[nz // 2]
        z_offset = z0_mm - z_mask_center
        z_mm = z_raw_mm + z_offset
        print(f"\nCoordinate mapping (legacy pixel->mm):")
        print(f"  X range: {x_mm.min():.2f} ~ {x_mm.max():.2f} mm")
        print(f"  Z range (raw): {z_raw_mm.min():.2f} ~ {z_raw_mm.max():.2f} mm")
        print(f"  Z offset to match focus: {z_offset:.2f} mm")
        print(f"  Z range (adjusted): {z_mm.min():.2f} ~ {z_mm.max():.2f} mm")
    XX, ZZ = np.meshgrid(x_mm, z_mm)
    # -------------------------------------------------------------------------
    # 3.5 Click to choose a point (override focus / sources)
    # -------------------------------------------------------------------------
    if args.click:
        # background to show: vessel intensity masked
        bg = (vint * mask.astype(np.float32))
        pt = pick_point_on_image(
            bg,
            x_mm=x_mm,
            z_mm=z_mm,
            title=f"Slice #{SLICE_IDX}: click source, Enter=save, Esc=cancel"
        )
        if pt is None:
            print("[CANCEL] No point selected.")
            return
        x0_mm, z0_mm = pt
        foci_x_mm = np.array([x0_mm], dtype=np.float32)
        foci_z_mm = np.array([z0_mm], dtype=np.float32)
        n_foci = 1
        print(f"[CLICK] Selected point: x={x0_mm:.3f} mm, z={z0_mm:.3f} mm")

        if args.single_point:
            # Save exactly one point source
            out_data = {
                "src_x_mm": np.array([x0_mm], dtype=np.float32),
                "src_z_mm": np.array([z0_mm], dtype=np.float32),
                "src_amp": np.array([float(args.amp)], dtype=np.float32),
                "src_type": np.array([int(args.src_type)], dtype=np.uint8),
                "foci_x_mm": foci_x_mm.astype(np.float32),
                "foci_z_mm": foci_z_mm.astype(np.float32),
                "x0_mm": np.float32(x0_mm),
                "z0_mm": np.float32(z0_mm),
                "n_foci": np.int32(n_foci),
                "sigma_mm": np.float32(sigma_mm),
                "pixel_um": np.float32(PIXEL_UM),
                "slice_idx": np.int32(SLICE_IDX),
            }
            savemat(OUT_PATH, out_data)
            print(f"\n=== Saved single-point source: {OUT_PATH} ===")
            return
    # -------------------------------------------------------------------------
    # 4. Focus 가중치 계산 (multi-foci 지원)
    # -------------------------------------------------------------------------
    # 각 foci에 대해 Gaussian weight 계산 후 합산
    w_focus = np.zeros_like(XX, dtype=np.float32)
    
    for i in range(n_foci):
        fx, fz = foci_x_mm[i], foci_z_mm[i]
        w_focus += np.exp(-((XX - fx)**2 + (ZZ - fz)**2) / (2 * sigma_mm**2))
    
    # 정규화
    w_focus = w_focus / (w_focus.max() + 1e-12)
    
    # -------------------------------------------------------------------------
    # 5. 최종 가중치 및 sampling
    # -------------------------------------------------------------------------
    # 가중치: vessel intensity × focus weight × mask
    w = (vint * w_focus) * mask
    
    idx_all = np.flatnonzero(mask)
    w_flat = w[mask].astype(np.float64)
    
    # NaN/Inf/음수 제거
    w_flat[~np.isfinite(w_flat)] = 0.0
    w_flat[w_flat < 0] = 0.0
    
    M = idx_all.size
    K = min(K_default, M)
    
    if M == 0:
        raise RuntimeError("Mask is empty! Check slice index or vessel data.")
    
    print(f"\nSampling {K} sources from {M} vessel pixels...")
    
    wsum = w_flat.sum()
    if wsum <= 0:
        # fallback: uniform sampling
        print("Warning: All weights are zero, using uniform sampling.")
        pick_in = np.random.choice(M, size=K, replace=False)
    else:
        p = w_flat / wsum
        p = np.clip(p, 0.0, 1.0)
        p = p / p.sum()  # 부동소수 오차 보정
        pick_in = np.random.choice(M, size=K, replace=False, p=p)
    
    pick_lin = idx_all[pick_in]
    iz = pick_lin // nx
    ix = pick_lin % nx
    
    # Source 좌표
    src_x_mm = x_mm[ix].astype(np.float32)
    src_z_mm = z_mm[iz].astype(np.float32)
    
    # Source 진폭 (혈관 반지름 기반)
    src_amp = radf[iz, ix].astype(np.float32)
    
    # -------------------------------------------------------------------------
    # 6. Stable vs Inertial 분류
    # -------------------------------------------------------------------------
    # Focus 중심에 가까울수록 inertial 비율 높음
    focus_strength = w_focus[iz, ix]
    focus_norm = (focus_strength - focus_strength.min()) / (focus_strength.max() - focus_strength.min() + 1e-12)
    prob_inertial = INERTIAL_BASE + (INERTIAL_MAX - INERTIAL_BASE) * focus_norm
    prob_inertial = np.clip(prob_inertial, 0, 1)
    
    src_type = (np.random.rand(K) < prob_inertial).astype(np.uint8)
    # 1 = inertial, 0 = stable
    
    n_inertial = src_type.sum()
    n_stable = K - n_inertial
    print(f"Source types: {n_stable} stable, {n_inertial} inertial")
    
    # -------------------------------------------------------------------------
    # 7. 저장
    # -------------------------------------------------------------------------
    out_data = {
        # Source 정보
        "src_x_mm": src_x_mm,
        "src_z_mm": src_z_mm,
        "src_amp": src_amp,
        "src_type": src_type,
        
        # FUS 설정 (P.mat에서 가져온 값)
        "foci_x_mm": foci_x_mm.astype(np.float32),
        "foci_z_mm": foci_z_mm.astype(np.float32),
        "x0_mm": np.float32(x0_mm),
        "z0_mm": np.float32(z0_mm),
        "n_foci": np.int32(n_foci),
        
        # 메타 정보
        "sigma_mm": np.float32(sigma_mm),
        "pixel_um": np.float32(PIXEL_UM),
        "slice_idx": np.int32(SLICE_IDX),
    }
    
    savemat(OUT_PATH, out_data)
    
    print(f"\n=== Saved: {OUT_PATH} ===")
    print(f"N sources: {K}")
    print(f"X range: {src_x_mm.min():.2f} ~ {src_x_mm.max():.2f} mm")
    print(f"Z range: {src_z_mm.min():.2f} ~ {src_z_mm.max():.2f} mm")
    print(f"Focus center: x={x0_mm:.2f} mm, z={z0_mm:.2f} mm")
    
    # PCI grid와 비교
    print(f"\n=== Sanity check vs PCI grid ===")
    z_in_grid = (src_z_mm >= aZ.min()*1000) & (src_z_mm <= aZ.max()*1000)
    x_in_grid = (src_x_mm >= aX.min()*1000) & (src_x_mm <= aX.max()*1000)
    in_grid = z_in_grid & x_in_grid
    print(f"Sources within PCI grid: {in_grid.sum()} / {K} ({in_grid.sum()/K*100:.1f}%)")
    
    if in_grid.sum() < K * 0.5:
        print("WARNING: Less than 50% of sources are within PCI grid!")
        print("Check vessel mask position or adjust z_offset.")


if __name__ == "__main__":
    main()

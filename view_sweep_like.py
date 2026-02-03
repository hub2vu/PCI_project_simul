import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="vessel_sweep_out", help="npy 폴더")
    ap.add_argument("--kind", default="vessel_intensity", choices=["vessel_intensity","radius_field_um","mask","dist_um"],
                    help="뭘 볼지")
    ap.add_argument("--idx0", type=int, default=0, help="시작 인덱스")
    ap.add_argument("--vmax_p", type=float, default=99.5, help="표시 vmax 퍼센타일(강한 밝기 클리핑)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.dir)

    patt = {
        "vessel_intensity": "vessel_intensity_*.npy",
        "radius_field_um": "radius_field_um_*.npy",
        "mask": "mask_*.npy",
        "dist_um": "dist_um_*.npy",
    }[args.kind]

    files = sorted(glob.glob(os.path.join(out_dir, patt)))
    if not files:
        raise FileNotFoundError(f"no files: {os.path.join(out_dir, patt)}")

    n = len(files)
    i = max(0, min(args.idx0, n-1))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis("off")

    img = np.load(files[i])
    if args.kind == "mask":
        im = ax.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=1)
    else:
        vmax = np.percentile(img, args.vmax_p)
        vmax = max(float(vmax), 1e-6)
        im = ax.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=vmax)

    title = ax.set_title(f"{i+1}/{n} | {os.path.basename(files[i])}")

    def show(k):
        nonlocal i
        i = max(0, min(k, n-1))
        arr = np.load(files[i])
        if args.kind == "mask":
            im.set_data(arr)
            im.set_clim(0, 1)
        else:
            vmax = np.percentile(arr, args.vmax_p)
            vmax = max(float(vmax), 1e-6)
            im.set_data(arr)
            im.set_clim(0, vmax)
        title.set_text(f"{i+1}/{n} | {os.path.basename(files[i])}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ["right", "d", " "]:
            show(i + 1)
        elif event.key in ["left", "a", "backspace"]:
            show(i - 1)
        elif event.key in ["home"]:
            show(0)
        elif event.key in ["end"]:
            show(n - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

if __name__ == "__main__":
    main()

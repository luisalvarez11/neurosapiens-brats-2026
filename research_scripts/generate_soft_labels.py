"""
generate_soft_labels.py
========================

Genera soft labels (WT) a partir de segmentaciones binarias/multi-clase BraTS-PEDs,
calibradas por paciente usando la anchura de transición real medida en
distance_profile.py (anchuras_294.csv).

Convención de distancia con signo (igual que distance_profile.py):
    d < 0  -> dentro del tumor
    d = 0  -> contorno
    d > 0  -> fuera

Soft label:
    p(d) = 1 / (1 + exp(d / s))          s = width_mm / (2 * ln(9))

  -> p ~ 1 dentro, p ~ 0 fuera, anchura 10-90% == width_mm del paciente.
  Si el paciente no tiene anchura fiable (contrast<0 en el CSV o no está en
  la tabla), se usa la mediana global de la cohorte como fallback.

Uso:
    python generate_soft_labels.py \
        --csv anchuras_294.csv \
        --seg-dir /ruta/a/segmentaciones \
        --out-dir /ruta/salida \
        --pattern "{pid}-seg.nii.gz"
"""
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import distance_transform_edt

LN9 = np.log(9.0)  # logit(0.9) - logit(0.1) para logística estándar


def signed_distance_mm(mask, spacing):
    """d<0 dentro, d>0 fuera, en mm (idéntico a distance_profile.py)."""
    inside = distance_transform_edt(mask, sampling=spacing)
    outside = distance_transform_edt(~mask, sampling=spacing)
    return outside - inside


def soft_label_from_distance(d, width_mm, d_clip=None):
    """Logística calibrada: anchura 10-90% == width_mm."""
    s = max(width_mm, 1e-3) / (2 * LN9)
    if d_clip is not None:
        d = np.clip(d, -d_clip, d_clip)
    # p(d) = 1 / (1 + exp(d/s)); estable numéricamente con clip previo
    return 1.0 / (1.0 + np.exp(np.clip(d / s, -50, 50)))


def load_width_table(csv_path):
    """Devuelve dict patient_id -> width_mm fiable, y la mediana global (fallback)."""
    df = pd.read_csv(csv_path)
    global_median = float(df.loc[df.contrast > 0, "transition_width_mm"].median())
    reliable = df[df.contrast > 0].set_index("patient_id")["transition_width_mm"]
    widths = reliable.to_dict()
    n_bad = (df.contrast <= 0).sum()
    print(f"[widths] {len(widths)}/{len(df)} pacientes con anchura fiable "
          f"(mediana global fallback = {global_median:.2f} mm, {n_bad} excluidos por contrast<=0)")
    return widths, global_median


def generate_soft_label(seg_path, out_path, width_mm, wt_labels=None, d_clip=20.0):
    """
    seg_path : NIfTI de segmentación (multi-clase o binaria).
    wt_labels: set de enteros que forman el Whole Tumor. Si None, WT = seg > 0
               (convención BraTS estándar: cualquier label>0 es tumor).
    """
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj)
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])

    mask = np.isin(seg, list(wt_labels)) if wt_labels else (seg > 0)
    if mask.sum() == 0:
        print(f"  [warn] {seg_path.name}: sin tumor, se omite (mascara vacía)")
        return None

    d = signed_distance_mm(mask, spacing)
    soft = soft_label_from_distance(d, width_mm, d_clip=d_clip).astype(np.float32)

    out_img = nib.Nifti1Image(soft, img.affine, img.header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))
    return soft


def run_cohort(csv_path, seg_dir, out_dir, pattern="{pid}.nii.gz",
                wt_labels=None, d_clip=20.0):
    widths, global_median = load_width_table(csv_path)
    seg_dir, out_dir = Path(seg_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    n_ok, n_missing = 0, []
    for pid in df.patient_id:
        seg_path = seg_dir / pattern.format(pid=pid)
        if not seg_path.exists():
            n_missing.append(pid)
            continue
        width_mm = widths.get(pid, global_median)
        out_path = out_dir / f"{pid}-softlabel.nii.gz"
        result = generate_soft_label(seg_path, out_path, width_mm,
                                      wt_labels=wt_labels, d_clip=d_clip)
        if result is not None:
            n_ok += 1

    print(f"\nGeneradas {n_ok} soft labels en {out_dir}")
    if n_missing:
        print(f"{len(n_missing)} segmentaciones no encontradas (revisa --pattern), "
              f"ej: {n_missing[:3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="anchuras_294.csv")
    ap.add_argument("--seg-dir", required=True, help="carpeta con las segmentaciones NIfTI")
    ap.add_argument("--out-dir", required=True, help="carpeta de salida para las soft labels")
    ap.add_argument("--pattern", default="{pid}.nii.gz",
                     help="patrón del nombre de fichero de seg, usando {pid}")
    ap.add_argument("--wt-labels", default=None,
                     help="lista de enteros separados por coma que forman el WT "
                          "(por defecto: cualquier label>0)")
    ap.add_argument("--d-clip", type=float, default=20.0,
                     help="distancia máxima (mm) a la que se satura la soft label")
    args = ap.parse_args()

    wt_labels = None
    if args.wt_labels:
        wt_labels = {int(x) for x in args.wt_labels.split(",")}

    run_cohort(args.csv, args.seg_dir, args.out_dir, args.pattern,
               wt_labels=wt_labels, d_clip=args.d_clip)

#!/usr/bin/env python3
"""
construir_submission_combinada.py
=================================

Construye la submission del WT COMBINADO sobre el validation oficial.

Base: predsVal_final (submission 9773553 = remapeo + CC, valores 0/1/2/3/4).
Modificacion: el WT se reemplaza por la fusion:
    WT_combinado = (TC_soft > thr) OR (WT_hard)

donde:
  - TC_soft  = canal 1 de las probabilidades del modelo soft (mejor borde)
  - WT_hard  = todo lo que la 9773553 marca como tumor (>0)

El resto (TC, ET, CC) se mantiene EXACTAMENTE como la 9773553, para que la
unica diferencia sea el WT y cualquier cambio en Synapse sea atribuible a el.

Etiquetas oficiales BraTS-PEDs: ET=1, NET=2, CC=3, ED=4.
El WT ampliado por el combinado que NO estaba en el hard se etiqueta como
edema (4), ya que es tejido WT fuera del core.
"""

import numpy as np, os, glob, nibabel as nib
from scipy import ndimage

HARD_DIR = "/workspace/predsVal_final"          # 9773553 (0/1/2/3/4)
SOFT_DIR = "/workspace/preds_soft_oficial"      # probabilidades del soft
OUT_DIR  = "/workspace/preds_combinado_final"
TC_SOFT_CH = 1        # canal del TC soft (verificar!)
THR_SOFT = 0.5

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    hard_files = sorted(glob.glob(os.path.join(HARD_DIR, "*.nii.gz")))
    print(f"Casos hard (base 9773553): {len(hard_files)}")

    n_ok, sin_soft = 0, []
    for hp in hard_files:
        cid = os.path.basename(hp).replace(".nii.gz", "")
        sp = os.path.join(SOFT_DIR, cid + ".npz")
        if not os.path.exists(sp):
            sin_soft.append(cid)
            # sin soft, copiar el hard tal cual (no perder el caso)
            img = nib.load(hp)
            nib.save(img, os.path.join(OUT_DIR, cid + ".nii.gz"))
            continue

        img = nib.load(hp)
        hard = np.asanyarray(img.dataobj).astype(np.uint8)   # 0/1/2/3/4

        # TC soft desde probabilidades
        prob = np.load(sp)["probabilities"]
        tc_soft = prob[TC_SOFT_CH]
        # alinear orientacion con el hard
        if tc_soft.shape != hard.shape:
            tc_soft = np.transpose(tc_soft, (2, 1, 0))
        tc_soft_mask = tc_soft > THR_SOFT

        # --- WT combinado ---
        wt_hard = hard > 0                       # todo el tumor del hard
        wt_comb = wt_hard | tc_soft_mask         # fusion

        # --- reconstruir etiquetas ---
        # Partimos del hard (que ya tiene ET=1, NET=2, CC=3, ED=4 bien colocados).
        # El combinado solo AÑADE tejido WT (donde tc_soft se extiende mas que el hard).
        # Ese tejido nuevo se etiqueta como edema (4), por ser WT fuera del core.
        out = hard.copy()
        nuevo_wt = wt_comb & (hard == 0)         # voxeles que el combinado añade
        out[nuevo_wt] = 4                        # edema

        # (opcional) limpiar: el tejido añadido muy disperso podria ser ruido.
        # filtrar componentes minusculas del edema añadido
        add_mask = nuevo_wt
        if add_mask.sum() > 0:
            lab, nlab = ndimage.label(add_mask)
            if nlab > 0:
                sizes = np.bincount(lab.ravel())
                keep = sizes > 50      # descarta motas < 50 vox
                keep[0] = False
                small = add_mask & ~keep[lab]
                out[small] = 0         # revertir las motas a fondo

        nib.save(nib.Nifti1Image(out, img.affine, img.header),
                 os.path.join(OUT_DIR, cid + ".nii.gz"))
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"  {n_ok} casos combinados...")

    print(f"\nCombinados: {n_ok}")
    if sin_soft:
        print(f"[!] {len(sin_soft)} casos sin soft (copiados del hard): {sin_soft[:5]}")

    # verificacion
    f0 = sorted(glob.glob(os.path.join(OUT_DIR, "*.nii.gz")))[0]
    print(f"\nEjemplo {os.path.basename(f0)}: valores {np.unique(np.asanyarray(nib.load(f0).dataobj))}")
    print("Debe contener 0/1/2/3/4 (ET/NET/CC/ED).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extraer_features_ed.py  (v2)
============================

Extrae features por voxel ED (out-of-fold) + label real, Y ADEMAS guarda:
  - coords  : indices (z,y,x) de cada voxel ED dentro de su caso
  - case_ids: id de caso por voxel
  - las segmentaciones OOF COMPLETAS por caso en /workspace/preds_oof_construidas
Esto permite al entrenador medir el EFECTO NETO del remapeo en las 5 regiones.

Features (16): p_wt, p_tc, p_et, p_tc_soft, p_cc,
  t1n_z, t1c_z, t2w_z, t2f_z, +medias 3x3x3,
  dist_tc_border(=dist al TC mas cercano), comp_size, dist_centroid
"""

import os, glob, numpy as np, nibabel as nib
from scipy import ndimage

HARD = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
SOFT = "/workspace/preds_soft_interno"
GT_DIR = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"
IMG_DIR = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
OOF_OUT = "/workspace/preds_oof_construidas"

TC_SOFT_CH = 1
THR_SOFT = 0.5
MIN_COMPONENT = 50
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC, MIN_CC = 0.90, 100

FEAT_NAMES = [
    "p_wt", "p_tc", "p_et", "p_tc_soft", "p_cc",
    "t1n_z", "t1c_z", "t2w_z", "t2f_z",
    "t1n_z_m3", "t1c_z_m3", "t2w_z_m3", "t2f_z_m3",
    "dist_tc_border", "comp_size", "dist_centroid",
]

os.makedirs(OOF_OUT, exist_ok=True)


def filtrar(mask, minv):
    if mask.sum() == 0: return mask
    lab, n = ndimage.label(mask)
    if n == 0: return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes > minv)[lab]


def zscore(vol):
    m = vol > 0
    z = np.zeros_like(vol, dtype=np.float32)
    if m.sum() > 0:
        z[m] = (vol[m] - vol[m].mean()) / (vol[m].std() + 1e-6)
    return z


def main():
    Xs, ys, groups, all_coords, all_cids = [], [], [], [], []
    ncases = 0
    for f in range(5):
        val_dir = f"{HARD}/fold_{f}/validation"
        for ef in sorted(glob.glob(os.path.join(val_dir, "*.npz"))):
            cid = os.path.basename(ef).replace(".npz", "")
            gt_p = os.path.join(GT_DIR, f"{cid}.nii.gz")
            if not os.path.exists(gt_p): continue
            gt_img = nib.load(gt_p)
            gt = np.asanyarray(gt_img.dataobj).astype(np.uint8)

            prob = np.load(ef)["probabilities"]
            if prob[0].shape != gt.shape and prob[0].shape == gt.T.shape:
                prob = np.transpose(prob, (0, 3, 2, 1))
            p_wt, p_tc, p_et = prob[0], prob[1], prob[2]

            p_tc_soft = np.zeros_like(p_wt)
            sp = os.path.join(SOFT, f"{cid}.npz")
            if os.path.exists(sp):
                ts = np.load(sp)["probabilities"][TC_SOFT_CH]
                if ts.shape != gt.shape: ts = np.transpose(ts, (2, 1, 0))
                p_tc_soft = ts

            imgs = {}
            for ch, name in zip(range(4), ["t1n", "t1c", "t2w", "t2f"]):
                ip = os.path.join(IMG_DIR, f"{cid}_{ch:04d}.nii.gz")
                imgs[name] = zscore(np.asanyarray(nib.load(ip).dataobj).astype(np.float32))
            imgs_m3 = {k: ndimage.uniform_filter(v, size=3) for k, v in imgs.items()}

            logit = COEF_T1C * imgs["t1c"] + COEF_T2W * imgs["t2w"] + INTERCEPT
            p_cc = 1.0 / (1.0 + np.exp(-logit))

            # construccion (igual que la submission) para obtener la seg OOF completa
            wt = filtrar((p_wt > 0.5) | (p_tc_soft > THR_SOFT), MIN_COMPONENT)
            tc = (p_tc > 0.5) & wt
            et = (p_et > 0.5) & tc
            pred_ed = wt & ~tc
            net = tc & ~et
            # CC
            cc = filtrar(net & (p_cc > THR_CC), MIN_CC)
            # seg completa remapeada: ET=1,NET=2,CC=3,ED=4
            seg = np.zeros(gt.shape, dtype=np.uint8)
            seg[pred_ed] = 4; seg[net] = 2; seg[cc] = 3; seg[et] = 1
            # guardar seg OOF completa (para el efecto neto)
            nib.save(nib.Nifti1Image(seg, gt_img.affine, gt_img.header),
                     os.path.join(OOF_OUT, f"{cid}.nii.gz"))

            if pred_ed.sum() == 0:
                ncases += 1; continue

            # features espaciales
            dist_tc = ndimage.distance_transform_edt(~tc) if tc.sum() > 0 else np.full(gt.shape, 99.0)
            lab_ed, _ = ndimage.label(pred_ed)
            comp_sizes = np.bincount(lab_ed.ravel()); comp_sizes[0] = 0
            comp_size_map = comp_sizes[lab_ed].astype(np.float32)
            if wt.sum() > 0:
                cz, cy, cx = ndimage.center_of_mass(wt)
                zz, yy, xx = np.indices(gt.shape)
                dist_cent = np.sqrt((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2).astype(np.float32)
            else:
                dist_cent = np.zeros(gt.shape, dtype=np.float32)

            idx = np.where(pred_ed)
            feats = np.stack([
                p_wt[idx], p_tc[idx], p_et[idx], p_tc_soft[idx], p_cc[idx],
                imgs["t1n"][idx], imgs["t1c"][idx], imgs["t2w"][idx], imgs["t2f"][idx],
                imgs_m3["t1n"][idx], imgs_m3["t1c"][idx], imgs_m3["t2w"][idx], imgs_m3["t2f"][idx],
                dist_tc[idx], comp_size_map[idx], dist_cent[idx],
            ], axis=1).astype(np.float32)
            labels = gt[idx].astype(np.uint8)
            coords = np.stack(idx, axis=1).astype(np.int16)   # (n,3) z,y,x

            Xs.append(feats); ys.append(labels)
            groups.append(np.full(len(labels), ncases, dtype=np.int32))
            all_coords.append(coords)
            all_cids.append(np.array([cid] * len(labels)))
            ncases += 1
            if ncases % 25 == 0:
                print(f"  {ncases} casos, {sum(len(x) for x in Xs):,} voxeles ED")

    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(groups)
    coords = np.concatenate(all_coords); cids = np.concatenate(all_cids)
    print(f"\nTotal: {len(y):,} voxeles ED de {ncases} casos")
    print("Distribucion de labels reales en los voxeles ED:")
    for c in [0,1,2,3,4]:
        print(f"  clase {c}: {(y==c).sum():,} ({100*(y==c).mean():.1f}%)")
    np.savez_compressed("/workspace/features_ed.npz",
                        X=X, y=y, groups=g, coords=coords, case_ids=cids,
                        feat_names=np.array(FEAT_NAMES))
    print("\nGuardado: /workspace/features_ed.npz")
    print(f"Segmentaciones OOF completas en: {OOF_OUT}")


if __name__ == "__main__":
    main()

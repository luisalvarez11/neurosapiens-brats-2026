import os, glob, gc
import numpy as np
import nibabel as nib
from scipy import ndimage

PREDS_VAL = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
IMAGES    = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
GT_DIR    = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"
CHANNELS  = ["0000", "0001", "0002", "0003"]

# Clasificador comparacion B (edema vs sano peritumoral)
COEF = np.array([-0.0024, -0.7011, 0.4016, 1.8768], dtype=np.float32)
INTERCEPT = -3.2951

# Rejilla: umbrales altos (conservadores) para no meter tejido sano en el WT
DILATACIONES = [3, 5, 8]
UMBRALES     = [0.85, 0.90, 0.95, 0.98]
MIN_VOXELS   = [50, 150]


def zscore_full(path):
    v = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    m = v > 0
    mu, s = v[m].mean(), v[m].std()
    z = np.zeros_like(v, dtype=np.float32)
    z[m] = (v[m] - mu) / (s + 1e-6)
    return z


def dice(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if gs == 0 and ps == 0: return np.nan
    if gs == 0 or ps == 0: return 0.0
    return 2.0 * np.logical_and(pred, gt).sum() / (ps + gs)


def surface(mask):
    if mask.sum() == 0: return mask
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def nsd(pred, gt, tol=1.0):
    ps, gs = pred.sum(), gt.sum()
    if gs == 0 and ps == 0: return np.nan
    if gs == 0 or ps == 0: return 0.0
    sp, sg = surface(pred), surface(gt)
    if sp.sum() == 0 or sg.sum() == 0: return 0.0
    dg = ndimage.distance_transform_edt(~sg)
    dp = ndimage.distance_transform_edt(~sp)
    close = (dg[sp] <= tol).sum() + (dp[sg] <= tol).sum()
    return close / (sp.sum() + sg.sum())


def main():
    files = sorted(glob.glob(os.path.join(PREDS_VAL, "*.nii.gz")))
    nD, nT, nM = len(DILATACIONES), len(UMBRALES), len(MIN_VOXELS)

    # acumular por caso -> luego promediar
    wt_d = np.full((0, nD, nT, nM), np.nan)
    ed_d = np.full((0, nD, nT, nM), np.nan)
    wtnsd_d = np.full((0, nD, nT, nM), np.nan)
    wt_ref_l, ed_ref_l, wtnsd_ref_l, net_ref_l = [], [], [], []

    # listas para apilar
    WT, ED, WTN = [], [], []

    n_ok = 0
    for pf in files:
        cid = os.path.basename(pf).replace(".nii.gz", "")
        gp = os.path.join(GT_DIR, cid + ".nii.gz")
        chp = [os.path.join(IMAGES, f"{cid}_{c}.nii.gz") for c in CHANNELS]
        if not (os.path.exists(gp) and all(os.path.exists(p) for p in chp)):
            continue
        n_ok += 1

        seg = np.asanyarray(nib.load(pf).dataobj).astype(np.uint8)
        gt  = np.asanyarray(nib.load(gp).dataobj).astype(np.uint8)
        wt_pred = (seg >= 1)
        brain = np.asanyarray(nib.load(chp[0]).dataobj).astype(np.float32) > 0

        gt_wt = (gt >= 1); gt_ed = (gt == 4); gt_net = (gt == 2)

        # referencia (sin edema anadido)
        wt_ref_l.append(dice(wt_pred, gt_wt))
        ed_ref_l.append(dice(np.zeros_like(seg, bool), gt_ed))
        wtnsd_ref_l.append(nsd(wt_pred, gt_wt))
        net_ref_l.append(dice((seg == 2) | (seg == 1), gt_net))

        # anillo maximo (dilatacion mayor) y sus probabilidades
        max_ring = ndimage.binary_dilation(wt_pred, iterations=max(DILATACIONES)) & ~wt_pred & brain
        idx = np.flatnonzero(max_ring.ravel())

        # logit por canal (bajo RAM)
        logit = np.full(idx.size, INTERCEPT, dtype=np.float32)
        for k, p in enumerate(chp):
            z = zscore_full(p).ravel()[idx]
            logit += COEF[k] * z
            del z; gc.collect()
        prob = 1.0 / (1.0 + np.exp(-logit))

        # matrices por caso
        cwt = np.full((nD, nT, nM), np.nan)
        ced = np.full((nD, nT, nM), np.nan)
        cwn = np.full((nD, nT, nM), np.nan)

        cand = np.zeros(seg.shape, dtype=bool)
        for di, dil in enumerate(DILATACIONES):
            ring_d = ndimage.binary_dilation(wt_pred, iterations=dil) & ~wt_pred & brain
            in_d = ring_d.ravel()[idx]   # cuales del anillo max estan en esta dilatacion
            for ti, thr in enumerate(UMBRALES):
                sel = in_d & (prob > thr)
                cand.fill(False)
                if sel.any():
                    cand.ravel()[idx[sel]] = True
                    lab, n = ndimage.label(cand)
                    sizes = np.bincount(lab.ravel()) if n > 0 else np.array([0])
                else:
                    lab, n, sizes = None, 0, None
                for mi, mv in enumerate(MIN_VOXELS):
                    if n > 0:
                        keep = sizes > mv; keep[0] = False
                        ed_add = keep[lab]
                    else:
                        ed_add = np.zeros(seg.shape, dtype=bool)
                    wt_final = wt_pred | ed_add
                    cwt[di, ti, mi] = dice(wt_final, gt_wt)
                    ced[di, ti, mi] = dice(ed_add, gt_ed)
                    cwn[di, ti, mi] = nsd(wt_final, gt_wt)
        WT.append(cwt); ED.append(ced); WTN.append(cwn)
        del seg, gt, wt_pred, brain, prob, logit, cand
        gc.collect()
        print(f"  procesados {n_ok} casos...", end="\r")

    WT = np.stack(WT); ED = np.stack(ED); WTN = np.stack(WTN)
    WT_REF = np.nanmean(wt_ref_l); ED_REF = np.nanmean(ed_ref_l)
    WTN_REF = np.nanmean(wtnsd_ref_l); NET_REF = np.nanmean(net_ref_l)

    print("\n\n=== REFERENCIA (sin edema) ===")
    print(f"  WT_dice={WT_REF:.4f}  WT_nsd={WTN_REF:.4f}  NETC={NET_REF:.4f}  ED={ED_REF:.4f}\n")

    print(f"{'dil':>4}{'thr':>6}{'minV':>6}{'WTdice':>9}{'dWT':>8}{'WTnsd':>8}{'dNSD':>8}{'ED':>8}")
    print("-" * 57)
    best = None
    for di, dil in enumerate(DILATACIONES):
        for ti, thr in enumerate(UMBRALES):
            for mi, mv in enumerate(MIN_VOXELS):
                mwt = np.nanmean(WT[:, di, ti, mi])
                mnsd = np.nanmean(WTN[:, di, ti, mi])
                med = np.nanmean(ED[:, di, ti, mi])
                dwt = mwt - WT_REF; dnsd = mnsd - WTN_REF
                flag = ""
                if dwt >= -0.002 and med > 0.02:
                    flag = "  <-- OK"
                    if best is None or med > best[-1]:
                        best = (dil, thr, mv, mwt, dwt, mnsd, med)
                print(f"{dil:>4}{thr:>6.2f}{mv:>6}{mwt:>9.4f}{dwt:>+8.4f}{mnsd:>8.4f}{dnsd:>+8.4f}{med:>8.4f}{flag}")
    print("-" * 57)
    if best:
        print(f"\nMEJOR (WT protegido): dil={best[0]} thr={best[1]} minVox={best[2]}")
        print(f"  WT_dice={best[3]:.4f} (dWT={best[4]:+.4f})  WT_nsd={best[5]:.4f}  ED={best[6]:.4f}")
    else:
        print("\nNinguna config recupera ED sin bajar el WT. Cerrar el edema.")


if __name__ == "__main__":
    main()

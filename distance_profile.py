"""
distance_profile.py
===================

EXPERIMENTO 1: perfil de señal vs distancia con signo a la cáscara del tumor.

Idea (no circular): las bandas se definen por DISTANCIA GEOMÉTRICA al contorno
del radiólogo (independiente de la intensidad). Dentro de cada banda medimos la
señal. Esto revela:
  - DÓNDE pica la transición de intensidad respecto al contorno (¿en d=0?).
  - La ANCHURA de la zona de transición (= anchura de infiltración), en mm:
    nítido -> estrecha; difuso/infiltrante -> ancha.

Convención de distancia con signo (mm):
  d < 0  -> DENTRO del tumor
  d = 0  -> contorno del radiólogo
  d > 0  -> FUERA (tejido peritumoral / sano)

Métricas por paciente:
  - transition_width_mm : anchura 10-90% de la transición de intensidad (biomarcador).
  - peak_offset_mm      : distancia donde el gradiente es máximo (dónde está el
                          borde de imagen respecto al contorno trazado).
  - contrast            : salto de intensidad dentro vs fuera (normalización).
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import (distance_transform_edt, binary_dilation,
                           binary_fill_holes, label as cc_label, sobel,
                           gaussian_filter)


def brain_mask_from_flair(flair):
    thr = np.quantile(flair[flair > 0], 0.25) if (flair > 0).any() else 0
    bm = binary_fill_holes(flair > thr)
    lab, n = cc_label(bm)
    if n > 1:
        s = np.bincount(lab.ravel()); s[0] = 0; bm = lab == s.argmax()
    return bm


def signed_distance_mm(mask, spacing):
    """d<0 dentro, d>0 fuera, en mm (usa el spacing del NIfTI)."""
    inside = distance_transform_edt(mask, sampling=spacing)
    outside = distance_transform_edt(~mask, sampling=spacing)
    return outside - inside        # >0 fuera, <0 dentro


def grad_mag(vol, spacing):
    g = [sobel(vol, axis=a, mode='reflect') / spacing[a] for a in range(3)]
    return np.sqrt(sum(gi**2 for gi in g)).astype(np.float32)


def radial_profile(flair, seg, spacing, d_max=12.0, bin_mm=1.0):
    """Devuelve (centros_bin, intensidad_media, gradiente_medio) vs distancia."""
    mask = seg > 0
    brain = brain_mask_from_flair(flair)
    d = signed_distance_mm(mask, spacing)
    g = grad_mag(gaussian_filter(flair, 0.5), spacing)

    region = brain & (np.abs(d) <= d_max)
    dd, ii, gg = d[region], flair[region], g[region]

    edges = np.arange(-d_max, d_max + bin_mm, bin_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    I = np.full(len(centers), np.nan)
    G = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = (dd >= edges[k]) & (dd < edges[k + 1])
        if sel.sum() >= 10:
            I[k] = np.median(ii[sel])
            G[k] = np.median(gg[sel])
    return centers, I, G


def extract_metrics(centers, I, G, d_max=12.0):
    """De los perfiles saca anchura de transición (10-90) y offset del pico de gradiente."""
    ok = ~np.isnan(I)
    c, Iv = centers[ok], I[ok]
    if len(c) < 5:
        return dict(transition_width_mm=np.nan, peak_offset_mm=np.nan, contrast=np.nan)
    # plateaus: dentro (d muy negativo) y fuera (d muy positivo)
    I_in = np.median(Iv[c <= c.min() + 2])
    I_out = np.median(Iv[c >= c.max() - 2])
    contrast = I_in - I_out
    if abs(contrast) < 1e-6:
        return dict(transition_width_mm=np.nan, peak_offset_mm=np.nan, contrast=float(contrast))
    # normalizar 1 (dentro) -> 0 (fuera)
    norm = (Iv - I_out) / (I_in - I_out)
    # cruces 0.9 (lado dentro) y 0.1 (lado fuera): anchura de transición
    def cross(level):
        for k in range(len(norm) - 1):
            a, b = norm[k], norm[k + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                t = (level - a) / (b - a)
                return c[k] + t * (c[k + 1] - c[k])
        return np.nan
    d90, d10 = cross(0.9), cross(0.1)
    width = abs(d10 - d90) if (np.isfinite(d90) and np.isfinite(d10)) else np.nan
    # offset del pico de gradiente
    okg = ~np.isnan(G)
    peak = centers[okg][np.argmax(G[okg])] if okg.any() else np.nan
    return dict(transition_width_mm=float(width), peak_offset_mm=float(peak),
                contrast=float(contrast))


def analyze_case(flair, seg, spacing, d_max=12.0):
    if (seg > 0).sum() < 50:
        return None
    centers, I, G = radial_profile(flair, seg, spacing, d_max)
    m = extract_metrics(centers, I, G, d_max)
    m['_profile'] = (centers, I, G)
    return m


# ----------------------- validación con anchura conocida -------------------
def phantom(size=80, transition_mm=2, contrast=0.6, seed=0, spacing=(1,1,1)):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    c = np.array([size/2]*3) + rng.uniform(-3, 3, 3)
    r = np.sqrt(((zz-c[0])**2 + (yy-c[1])**2 + (xx-c[2])**2))
    R = size*0.3; brain = r < size*0.46
    vol = np.where(brain, 0.5 + 0.05*rng.standard_normal((size,)*3), 0).astype(np.float32)
    t = transition_mm
    ramp = np.clip(1.0 - (r - (R - t)) / (2*t + 1e-3), 0, 1)   # transición ±t alrededor de R
    vol += contrast * ramp * brain
    seg = (r < R).astype(np.int32)
    return vol.astype(np.float32), seg


if __name__ == "__main__":
    from scipy.stats import spearmanr
    print("Validación: ¿la anchura medida recupera la anchura real de transición?\n")
    print(f"{'transición real (±mm)':<24}{'anchura 10-90 medida':<22}{'pico grad (mm)'}")
    print("-" * 60)
    truths, measured = [], []
    for t in [1, 2, 3, 4, 6, 8]:
        ws, ps = [], []
        for s in range(4):
            vol, seg = phantom(80, transition_mm=t, seed=s)
            m = analyze_case(vol, seg, spacing=(1,1,1))
            ws.append(m['transition_width_mm']); ps.append(m['peak_offset_mm'])
        truths.append(t); measured.append(np.nanmean(ws))
        print(f"{t:<24}{np.nanmean(ws):<22.2f}{np.nanmean(ps):+.2f}")
    rho, p = spearmanr(truths, measured)
    print(f"\nSpearman(anchura real, anchura medida) = {rho:+.3f} (p={p:.1e})")
    print("Pico de gradiente ~0 mm: el contorno coincide con la transición de imagen "
          "(fantasma sin margen, como BraTS).")


# ----------------------- cohorte + figura ----------------------------------
def run_cohort_nifti(pairs):
    """pairs: lista de (patient_id, flair_path, seg_path). Devuelve (rows, profiles)."""
    rows, profiles = [], {}
    for pid, fp, sp in pairs:
        fimg = nib.load(fp)
        flair = fimg.get_fdata().astype(np.float32)
        seg = nib.load(sp).get_fdata().astype(np.int32)
        spacing = tuple(float(x) for x in fimg.header.get_zooms()[:3])
        m = analyze_case(flair, seg, spacing)
        if m is None:
            continue
        profiles[pid] = m.pop('_profile')
        m['patient_id'] = pid
        rows.append(m)
    return rows, profiles


def make_figure(rows, profiles, out_path="distance_profile_fig.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    widths = {r['patient_id']: r['transition_width_mm'] for r in rows
              if np.isfinite(r['transition_width_mm'])}
    if not widths:
        print("sin anchuras válidas"); return
    ids_sorted = sorted(widths, key=widths.get)
    sharp, diff = ids_sorted[0], ids_sorted[-1]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) perfiles de intensidad normalizados: nítido vs difuso
    for pid, col, lab in ((sharp, 'tab:blue', f'más nítido ({widths[sharp]:.1f} mm)'),
                          (diff, 'tab:red', f'más difuso ({widths[diff]:.1f} mm)')):
        c, I, G = profiles[pid]
        ok = ~np.isnan(I)
        cc, Iv = c[ok], I[ok]
        I_in = np.median(Iv[cc <= cc.min()+2]); I_out = np.median(Iv[cc >= cc.max()-2])
        norm = (Iv - I_out)/(I_in - I_out + 1e-8)
        ax[0].plot(cc, norm, color=col, lw=2, label=lab)
    ax[0].axvline(0, color='k', ls='--', lw=1, label='contorno radiólogo')
    ax[0].axhspan(0.1, 0.9, color='gray', alpha=0.08)
    ax[0].set_xlabel('distancia con signo a la cáscara (mm)\n← dentro    fuera →')
    ax[0].set_ylabel('intensidad FLAIR normalizada')
    ax[0].set_title('Perfil de transición: nítido vs difuso'); ax[0].legend(fontsize=8)

    # (2) histograma de anchuras de transición en la cohorte
    w = np.array(list(widths.values()))
    ax[1].hist(w, bins=8, color='seagreen', alpha=0.8, edgecolor='k')
    ax[1].axvline(np.median(w), color='red', lw=1.5, label=f'mediana {np.median(w):.1f} mm')
    ax[1].set_xlabel('anchura de transición (mm)'); ax[1].set_ylabel('nº pacientes')
    ax[1].set_title('Anchura de infiltración en la cohorte'); ax[1].legend(fontsize=8)

    # (3) offset del pico de gradiente (¿el contorno está sobre la transición?)
    off = np.array([r['peak_offset_mm'] for r in rows if np.isfinite(r['peak_offset_mm'])])
    ax[2].hist(off, bins=8, color='slateblue', alpha=0.8, edgecolor='k')
    ax[2].axvline(0, color='k', ls='--', lw=1, label='sobre el contorno')
    ax[2].set_xlabel('offset del pico de gradiente (mm)\n← dentro    fuera →')
    ax[2].set_ylabel('nº pacientes')
    ax[2].set_title('¿Dónde está el borde de imagen?'); ax[2].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig); print("figura ->", out_path)


def _demo():
    # cohorte sintética mixta (nítidos y difusos) para probar figura/cohorte
    rows, profiles = [], {}
    plan = [('S1',1),('S2',2),('S3',2),('M1',4),('M2',5),('D1',7),('D2',8),('D3',9)]
    for pid, t in plan:
        vol, seg = phantom(80, transition_mm=t, seed=hash(pid) % 100)
        m = analyze_case(vol, seg, spacing=(1,1,1))
        profiles[pid] = m.pop('_profile'); m['patient_id'] = pid; rows.append(m)
    return rows, profiles


if __name__ == "__main__" and True:
    print("\n--- demo cohorte sintética: figura + tabla ---")
    rows, profiles = _demo()
    for r in rows:
        print(f"  {r['patient_id']}: anchura={r['transition_width_mm']:.2f} mm  "
              f"pico={r['peak_offset_mm']:+.2f} mm")
    make_figure(rows, profiles, "/home/claude/distance_profile_fig.png")

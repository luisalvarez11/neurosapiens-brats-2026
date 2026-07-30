"""
test_soft_labels.py
====================

Validación de generate_soft_labels.py en dos niveles:

1. TESTS AUTOMÁTICOS (geometría sintética con respuesta conocida):
   - rango de valores en [0,1]
   - p(contorno) == 0.5
   - monotonía: p decrece según te alejas hacia fuera
   - la anchura 10-90% medida en la soft label reproduce la width_mm pedida
   - threshold a 0.5 recupera la máscara binaria original (Dice ~ 1)
   - simetría: invertir dentro/fuera no debería romper nada

2. INSPECCIÓN VISUAL (para un caso real, opcional):
   - guarda una figura con: máscara dura, soft label, y perfil radial 1D
     comparado contra la curva logística teórica -> permite ver a ojo si
     algo no cuadra (ej. mask/seg desalineados, spacing mal leído, etc.)

Uso:
    python test_soft_labels.py                     # solo tests sintéticos
    python test_soft_labels.py --seg caso.nii.gz --width 5.3 --out check.png
"""
import sys
import argparse
import numpy as np
from generate_soft_labels import signed_distance_mm, soft_label_from_distance


# --------------------------- helpers de test --------------------------- #

def sphere_mask(size=60, radius=20, spacing=(1, 1, 1)):
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    c = size / 2
    r = np.sqrt(((zz - c) * spacing[0])**2 +
                ((yy - c) * spacing[1])**2 +
                ((xx - c) * spacing[2])**2)
    return r < radius, r


def measured_width_1090(d_line, p_line):
    order = np.argsort(d_line)
    dd, pp = d_line[order], p_line[order]

    def cross(level):
        for k in range(len(pp) - 1):
            a, b = pp[k], pp[k + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                t = (level - a) / (b - a)
                return dd[k] + t * (dd[k + 1] - dd[k])
        return np.nan
    d90, d10 = cross(0.9), cross(0.1)
    return abs(d10 - d90) if np.isfinite(d90) and np.isfinite(d10) else np.nan


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = (a & b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-9)


# ------------------------------- tests ---------------------------------- #

def test_range_and_center(width_mm=5.0):
    mask, r = sphere_mask()
    d = signed_distance_mm(mask, (1, 1, 1))
    soft = soft_label_from_distance(d, width_mm)
    assert soft.min() >= 0.0 and soft.max() <= 1.0, "soft label fuera de [0,1]"
    center_val = soft[30, 30, 30]  # centro de la esfera, muy dentro
    assert center_val > 0.99, f"centro debería ser ~1, es {center_val:.3f}"
    print(f"  [OK] rango en [0,1], centro={center_val:.4f}")


def test_boundary_is_half(width_mm=5.0):
    """En una rejilla discreta ningún voxel cae exactamente en d=0, así que
    interpolamos el cruce p=0.5 a lo largo de una línea radial y comprobamos
    que ocurre en d~0 (comparar voxeles contra 0.5 con |d|<eps daría un
    falso positivo con un filtro vacío, como pasaba antes)."""
    mask, r = sphere_mask()
    d = signed_distance_mm(mask, (1, 1, 1))
    soft = soft_label_from_distance(d, width_mm)
    line_d, line_p = d[30, 30, :], soft[30, 30, :]
    order = np.argsort(line_d)
    dd, pp = line_d[order], line_p[order]
    crossing = None
    for k in range(len(pp) - 1):
        a, b = pp[k], pp[k + 1]
        if (a - 0.5) * (b - 0.5) <= 0 and a != b:
            t = (0.5 - a) / (b - a)
            crossing = dd[k] + t * (dd[k + 1] - dd[k])
            break
    assert crossing is not None, "no se encontró cruce p=0.5 en la línea radial"
    assert abs(crossing) < 1.0, f"cruce p=0.5 debería estar en d~0, está en d={crossing:.3f}"
    print(f"  [OK] cruce p=0.5 en d={crossing:.3f}mm (esperado ~0)")


def test_monotonic(width_mm=5.0):
    mask, r = sphere_mask()
    d = signed_distance_mm(mask, (1, 1, 1))
    soft = soft_label_from_distance(d, width_mm)
    line_d = d[30, 30, :]
    line_p = soft[30, 30, :]
    order = np.argsort(line_d)
    diffs = np.diff(line_p[order])
    assert (diffs <= 1e-6).all(), "la soft label no es monótona decreciente con d"
    print("  [OK] monotonía dentro->fuera correcta")


def test_width_calibration(widths=(2.0, 5.0, 10.0, 15.0), tol_rel=0.15):
    mask, r = sphere_mask(size=80, radius=30)
    d = signed_distance_mm(mask, (1, 1, 1))
    for w in widths:
        soft = soft_label_from_distance(d, w)
        line_d, line_p = d[40, 40, :], soft[40, 40, :]
        measured = measured_width_1090(line_d, line_p)
        rel_err = abs(measured - w) / w
        status = "OK" if rel_err < tol_rel else "FALLO"
        print(f"  [{status}] width pedida={w:5.1f}mm  medida={measured:5.2f}mm  "
              f"error={rel_err*100:4.1f}%")
        assert rel_err < tol_rel, f"calibración de anchura fuera de tolerancia ({w}mm)"


def test_threshold_recovers_mask(width_mm=3.0):
    mask, r = sphere_mask()
    d = signed_distance_mm(mask, (1, 1, 1))
    soft = soft_label_from_distance(d, width_mm)
    hard_back = soft >= 0.5
    score = dice(mask, hard_back)
    assert score > 0.99, f"Dice(mask original, threshold 0.5) = {score:.4f}, debería ser ~1"
    print(f"  [OK] Dice(mask, threshold@0.5) = {score:.4f}")


def test_dclip_saturates():
    mask, r = sphere_mask()
    d = signed_distance_mm(mask, (1, 1, 1))
    soft = soft_label_from_distance(d, width_mm=3.0, d_clip=10.0)
    far_outside = d > 15
    far_inside = d < -15
    assert (soft[far_outside] < 0.01).all(), "no satura a 0 lejos por fuera"
    assert (soft[far_inside] > 0.99).all(), "no satura a 1 lejos por dentro"
    print("  [OK] saturación correcta lejos del contorno")


def run_all_tests():
    print("=== Tests automáticos (geometría sintética) ===")
    tests = [test_range_and_center, test_boundary_is_half, test_monotonic,
             test_width_calibration, test_threshold_recovers_mask, test_dclip_saturates]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except AssertionError as e:
            print(f"  [FALLO] {e}")
            failed += 1
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"TODOS los tests pasaron ({len(tests)}/{len(tests)})")
    else:
        print(f"{failed}/{len(tests)} tests FALLARON — revisa generate_soft_labels.py")
    return failed == 0


# ------------------------ inspección visual (real) ----------------------- #

def visual_check(seg_path, width_mm, out_path="check.png"):
    import nibabel as nib
    from generate_soft_labels import signed_distance_mm as sd, soft_label_from_distance as sl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = nib.load(seg_path)
    seg = np.asarray(img.dataobj)
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    mask = seg > 0
    if mask.sum() == 0:
        print("segmentación vacía, no se puede comprobar"); return

    d = sd(mask, spacing)
    soft = sl(d, width_mm)

    zc = int(np.average(np.where(mask)[2], weights=None)) if mask.any() else seg.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].imshow(mask[:, :, zc].T, cmap='gray', origin='lower')
    ax[0].set_title(f'Máscara dura (slice z={zc})')
    im = ax[1].imshow(soft[:, :, zc].T, cmap='inferno', origin='lower', vmin=0, vmax=1)
    ax[1].contour(mask[:, :, zc].T, levels=[0.5], colors='cyan', linewidths=1)
    ax[1].set_title('Soft label (contorno original en cian)')
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    # perfil radial real vs curva teórica
    ys, xs = np.where(mask[:, :, zc])
    if len(ys):
        cy, cx = ys.mean(), xs.mean()
        yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
        rr = np.sqrt((yy - cy)**2 * spacing[0]**2 + (xx - cx)**2 * spacing[1]**2)
        dd = d[:, :, zc].ravel()
        pp = soft[:, :, zc].ravel()
        order = np.argsort(dd)
        ax[2].scatter(dd[order][::37], pp[order][::37], s=4, alpha=0.3, label='datos (voxeles)')
        dth = np.linspace(dd.min(), dd.max(), 200)
        s = width_mm / (2 * np.log(9))
        pth = 1 / (1 + np.exp(dth / s))
        ax[2].plot(dth, pth, 'r-', lw=2, label=f'logística teórica (w={width_mm:.1f}mm)')
        ax[2].axvline(0, color='k', ls='--', lw=1)
        ax[2].set_xlabel('distancia con signo (mm)')
        ax[2].set_ylabel('soft label')
        ax[2].set_title('Perfil real vs. teórico')
        ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura de comprobación -> {out_path}\n"
          f"Revisa: (1) el contorno cian coincide con el borde del blob naranja/amarillo, "
          f"(2) los puntos azules siguen la curva roja.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=None, help="NIfTI de segmentación real (opcional)")
    ap.add_argument("--width", type=float, default=5.0, help="width_mm para ese paciente")
    ap.add_argument("--out", default="check.png")
    args = ap.parse_args()

    ok = run_all_tests()

    if args.seg:
        print(f"\n=== Inspección visual: {args.seg} ===")
        visual_check(args.seg, args.width, args.out)

    sys.exit(0 if ok else 1)

import nibabel as nib
import numpy as np

# ==== Paso 1 — Inspecciona los vóxeles de tu predicción ====
pred = nib.load('/workspace/preds_combinada_final2/BraTS-PED-00267-000.nii.gz').get_fdata()
labels, counts = np.unique(pred, return_counts=True)
print("=== PASO 1: Labels en tu predicción ===")
for l, c in zip(labels, counts):
    print(f"Label {l}: {int(c)} vóxeles")

# ==== Paso 2 — Compara contra el ground truth del mismo caso ====
# AJUSTA esta ruta a donde tengas el GT del caso 00267
gt = nib.load('RUTA_GT/BraTS-PED-00267-000_seg.nii.gz').get_fdata()
labels_gt, counts_gt = np.unique(gt, return_counts=True)
print("\n=== PASO 2: Labels en el ground truth ===")
for l, c in zip(labels_gt, counts_gt):
    print(f"GT Label {l}: {int(c)} vóxeles")

# ==== Paso 3 — Overlap real (Dice) para aislar el canal ED ====
label_ed = 2  # AJUSTA si tu convención de ED no es el label 2
pred_ed = (pred == label_ed)
gt_ed = (gt == label_ed)

intersection = np.logical_and(pred_ed, gt_ed).sum()
dice = 2 * intersection / (pred_ed.sum() + gt_ed.sum() + 1e-8)
print("\n=== PASO 3: Dice local de ED ===")
print(f"Dice ED local: {dice:.4f}")
print(f"Vóxeles pred ED: {pred_ed.sum()}, Vóxeles GT ED: {gt_ed.sum()}")

# ==== Paso 4 — Derivación por resta WT - TC ====
wt = (pred == 2) | (pred == 1) | (pred == 3)  # AJUSTA según tus labels reales
tc = (pred == 1) | (pred == 3)
ed_derivado = wt & ~tc
print("\n=== PASO 4: ED derivado por resta (WT-TC) ===")
print(f"Vóxeles ED derivado: {ed_derivado.sum()}")
print(f"Vóxeles ED guardado (label {label_ed}): {pred_ed.sum()}")
print(f"¿Coinciden derivado y guardado? {np.array_equal(ed_derivado, pred_ed)}")
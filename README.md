# Docker BraTS-PEDs 2026 Task 2 — NeuroSapiens

Contenedor de inferencia que replica la submission **9774214** (ensemble 5-fold +
WT combinado + CC logistica).

## Contenido
- `Dockerfile` — imagen con CUDA 12.8, PyTorch 2.8, nnU-Net 2.8.1.
- `predict.py` — inferencia: lee `/input`, escribe `/output` plano.
- `preparar_modelos.sh` — copia los pesos necesarios a `./models/`.
- `README.md` — este fichero.

## Pipeline que ejecuta
1. Lee cada carpeta de caso en `/input` (4 modalidades: t1n, t1c, t2w, t2f).
2. Inferencia con el **ensemble 5-fold** (cascada 502→501).
3. WT combinado = WT_hard OR (TC_soft > 0.5)  [si el modelo soft esta incluido].
4. Remapeo a etiquetas oficiales: ET=1, NET=2, CC=3, ED=4.
5. CC: logistica de intensidad (T1c, T2w) con umbral 0.90.
6. Escribe `{caseID}.nii.gz` en `/output` (estructura plana).

---

## PASO 1 — Preparar los pesos (en el pod, donde estan los modelos)

```bash
cd /ruta/al/docker_bratsped
bash preparar_modelos.sh
```

Esto crea `./models/` con solo los `checkpoint_final.pth` de cada fold
(~1.3 GB en vez de 13 GB) y `./trainers/` con los trainers custom.

## PASO 2 — Construir la imagen

```bash
docker build -t bratsped-neurosapiens:latest .
```

## PASO 3 — Probar LOCALMENTE (imprescindible antes de subir)

El challenge NO verifica los contenedores hasta cerrar las colas. Un Docker que
falla = 0. Probadlo con el validation oficial:

```bash
# preparar carpetas de prueba
mkdir -p /tmp/test_out

# ejecutar EXACTAMENTE como el challenge (network none, input read-only, limites)
docker run \
  --rm \
  --network none \
  --gpus=all \
  --volume /workspace/Dataset_Validation_Oficial:/input:ro \
  --volume /tmp/test_out:/output:rw \
  --memory=48G --shm-size=16G \
  bratsped-neurosapiens:latest
```

### Verificar la salida
```bash
# debe haber un .nii.gz por caso, en estructura PLANA (sin subcarpetas)
ls /tmp/test_out/ | head
ls /tmp/test_out/*.nii.gz | wc -l    # debe coincidir con el numero de casos

# valores correctos (0,1,2,3,4) y WT no saturado
python3 -c "
import nibabel as nib, numpy as np, glob
f=sorted(glob.glob('/tmp/test_out/*.nii.gz'))[0]
d=np.asanyarray(nib.load(f).dataobj)
print('valores:', np.unique(d), '| WT vox:', (d>0).sum())
"
```

**Checklist de validacion:**
- [ ] Un `.nii.gz` por caso en `/output`, estructura plana (sin subcarpetas).
- [ ] Valores exactamente {0,1,2,3,4}.
- [ ] WT no saturado (decenas de miles de voxeles, no millones).
- [ ] El run NO intento escribir en `/input` (no crashea con read-only).
- [ ] Termino en un tiempo razonable (muy por debajo de 12h).

### Comparar con la submission conocida (opcional pero recomendado)
Si el output coincide con lo que subisteis en 9774214, el Docker es correcto:
```bash
python3 -c "
import nibabel as nib, numpy as np, glob, os
a='/tmp/test_out'; b='/workspace/preds_final_ensemble'  # la 9774214 local
difs=0
for f in sorted(glob.glob(a+'/*.nii.gz'))[:10]:
    cid=os.path.basename(f); bf=os.path.join(b,cid)
    if not os.path.exists(bf): continue
    da=np.asanyarray(nib.load(f).dataobj); db=np.asanyarray(nib.load(bf).dataobj)
    difs+=(da!=db).sum()
print('voxeles distintos en 10 casos:', difs, '(idealmente ~0)')
"
```

## PASO 4 — Subir a Synapse

```bash
# etiquetar para el registro de Synapse (PROJECT_ID de vuestro equipo)
docker tag bratsped-neurosapiens:latest docker.synapse.org/PROJECT_ID/bratsped-neurosapiens:latest

# login (usar token de Synapse)
docker login docker.synapse.org

# subir
docker push docker.synapse.org/PROJECT_ID/bratsped-neurosapiens:latest
```

Luego, en la web de Synapse: Task 2 → Submission → seleccionar la imagen subida.

## RECORDATORIOS CRITICOS
- **El short paper en OpenReview debe estar enviado** — solo evaluan Docker
  vinculado a un short paper.
- El nombre del equipo en Synapse debe coincidir.
- Estructura de `/output` PLANA (sin subcarpetas), o se invalida.
- Nunca escribir en `/input`.

## Notas
- Si quereis el Docker SOLO con el ensemble (sin soft), borrad la carpeta
  `models/Dataset503_BraTSPED/` antes del build; `predict.py` detecta su
  ausencia y usa solo el WT hard automaticamente.
- El ensemble de 5 folds multiplica el tiempo de inferencia x5 por caso.
  Con ~91-100 casos de test y A10G, deberia estar muy por debajo de 12h,
  pero vigilad el tiempo en la prueba local.

#!/bin/bash
set -euo pipefail

# 1. Autenticación automática con el Token
export SYNAPSE_AUTH_TOKEN="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4Mzk3MjMxNiwiaWF0IjoxNzgzOTcyMzE2LCJqdGkiOiI0MjA2NiIsInN1YiI6IjM1OTExMTYifQ.ez3AtxP0fl6OZYXXiGXGGJJ1x-jG7yKdiZq4yfRA6sCXQ55fROf5AkcDCpCwO5-BAWL2V47a8w9MxjjDOBE4Ri9UnsqX2HgWbUbWTp-LtcWdskstH8-UKIZqxCrrbulTPjvGcv4pph4rjEuGO9sO6eOhfiUBSRdLL86kXEvY9XbDf0V1e0tDcrjRfaGuXLoRqC3CBfglZY5ymRyImNJuFhB56ktoKHq8EOhYB9XdyJs7ouUxOcDPVeYVwlDKbNvmPIvXAhMf4bFdQy9aE1hYY_DeGGAVHejDwO0SOgW8GoT_qMyF7Ye3TTtNje3HfLqjZkJTD-qL5Hn-e_HtrEU6mA"

echo "=== 1. Creando las carpetas base ==="
mkdir -p /workspace/descargas_brats
mkdir -p /workspace/datos_descomprimidos

echo "=== 2. Iniciando descarga de los 3 bloques de BraTS ==="
echo "Descargando Training 1 (syn74837563)..."
synapse get syn74837563 --downloadLocation /workspace/descargas_brats

echo "Descargando Training 2 (syn74916879)..."
synapse get syn74916879 --downloadLocation /workspace/descargas_brats

echo "Descargando Validation (syn74837589)..."
synapse get syn74837589 --downloadLocation /workspace/descargas_brats

echo "=== 3. Descargas completadas. Iniciando descompresión ==="
for zip_file in /workspace/descargas_brats/*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Descomprimiendo $zip_file ..."
        unzip -q -o "$zip_file" -d /workspace/datos_descomprimidos/
    fi
done

echo "=== ¡Proceso finalizado! ==="
echo "Los zips originales están en: /workspace/descargas_brats"
echo "Las imágenes extraídas están en: /workspace/datos_descomprimidos"

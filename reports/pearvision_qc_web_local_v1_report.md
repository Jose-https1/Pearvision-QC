# PearVision QC - Web Local Dashboard V1 — Reporte

**Fecha:** 2026-05-21  
**Versión:** pearvision_qc_web_local_v1  
**Base:** pipeline completo de V6 (segmentación, bloqueo pre-U3, U3, política)

---

## 1. Qué se ha creado

### Archivos nuevos

| Archivo | Descripción |
|---|---|
| `scripts/pearvision_qc_web_local_v1.py` | Script principal del servidor web local |
| `reports/pearvision_qc_web_local_v1_report.md` | Este reporte |

### Dependencias instaladas (en .venv vía uv)

```
fastapi==0.136.1
uvicorn==0.47.0
```

Comando usado:
```powershell
uv add fastapi uvicorn
```

### Qué NO se ha modificado

- `scripts/pearvision_qc_realtime_camera_pro_v6.py` — intacto
- `scripts/analyze_quality.py` — intacto  
- Modelos `.pt` — intactos  
- `quality_rules.yaml` — intacto  
- U2, U3 — intactos  
- `.venv` — solo se añadieron fastapi y uvicorn

---

## 2. Arquitectura del sistema

```
Portátil
  └── pearvision_qc_web_local_v1.py
        ├── hilo de cámara (camera_loop)
        │     OpenCV -> detección V6 -> U3 -> política V6
        │     actualiza estado global compartido (_state)
        ├── FastAPI + Uvicorn (hilo principal, puerto 8000)
        │     GET  /              → HTML del dashboard
        │     GET  /video_feed    → stream MJPEG con overlay
        │     GET  /api/status    → JSON con última predicción
        │     GET  /api/health    → JSON de salud del servidor
        │     POST /api/save      → guarda evidencia en disco
        └── outputs/live_camera_qc_web_v1/
              frames_original/, frames_overlay/, masks/,
              roi_processed/, metadata/, live_predictions.csv

Móvil / tablet / otro PC
  └── Navegador → http://<IP_LAN>:8000
        ├── Vídeo MJPEG en tiempo real (cámara del portátil)
        ├── Resultado grande: PASA / REVISAR / RECHAZA / SIN PERA / MALA CAPTURA
        ├── Panel de datos técnicos actualizado cada 400 ms
        └── Botón "Guardar evidencia" → POST /api/save
```

---

## 3. Cómo ejecutarlo

### Comando básico

```powershell
cd "Desktop\Sistemas de Percepción y Visión Artificial\computer vision"
.venv\Scripts\python.exe scripts\pearvision_qc_web_local_v1.py --camera 0 --host 0.0.0.0 --port 8000 --infer-every 5
```

### Salida esperada al arrancar

```
============================================================
  PearVision QC - Web Local Dashboard V1
============================================================
  URL local (portátil):  http://127.0.0.1:8000
  URL LAN (móvil):       http://192.168.x.x:8000

  Para obtener tu IP LAN: ejecuta 'ipconfig' en PowerShell
  y busca 'Dirección IPv4' en el adaptador WiFi.
  Ejemplo para móvil:    http://TU_IP_LOCAL:8000

  El móvil debe estar en la misma red WiFi que el portátil.
  Si no conecta: revisar firewall de Windows, puerto 8000.
============================================================
```

### Parámetros disponibles

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `--camera` | `0` | Índice de cámara OpenCV |
| `--host` | `0.0.0.0` | Bind address (0.0.0.0 = todas las interfaces) |
| `--port` | `8000` | Puerto del servidor |
| `--infer-every` | `5` | Ejecutar inferencia cada N frames |

---

## 4. Cómo entrar desde el móvil

1. **Conectar el móvil a la misma red WiFi** que el portátil.
2. En el portátil, abrir PowerShell y ejecutar `ipconfig`.  
   Buscar la sección **"Adaptador de LAN inalámbrica Wi-Fi"** → **"Dirección IPv4"**.  
   Ejemplo: `192.168.1.105`
3. En el navegador del móvil escribir:  
   **`http://192.168.1.105:8000`**  
   (sustituyendo por tu IP real)
4. La interfaz carga automáticamente con el vídeo en directo.

---

## 5. Descripción de la interfaz

### Zona de vídeo (izquierda)
- Stream MJPEG en tiempo real desde la cámara del portátil.
- Overlay con contorno de la pera detectada, bounding box y etiqueta de decisión.
- Info en el borde inferior: FPS, p_good, p_bad, gate_reason.

### Panel derecho
- **Resultado grande** con color según decisión:
  - `PASA` → verde `#28c83c`
  - `REVISAR` → naranja `#ff8c00`
  - `RECHAZA` → rojo `#e02020`
  - `SIN PERA` → azul `#3a8fff`
  - `MALA CAPTURA` → amarillo-naranja `#e0a020`
- **Botón "Guardar evidencia"** → POST `/api/save` → guarda el frame actual con todos los archivos de V6.
- **Panel técnico** con: capture_status, mask_valid, stable_decision, instant_decision, u3_pred, p_good, p_bad, threshold_good, threshold_bad, pear_area_ratio, bbox, gate_reason, FPS, latencia, saved_count, last_saved.
- **Barra de probabilidad** (verde=p_good, roja=p_bad) con línea de umbral.

### Endpoints API

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/video_feed` | GET | Stream MJPEG |
| `/api/status` | GET | JSON con última predicción completa |
| `/api/health` | GET | JSON `{"status":"ok","camera_alive":bool,"frame_id":int}` |
| `/api/save` | POST | Guarda evidencia, responde `{"ok":true,"name":"..."}` |

---

## 6. Qué hacer si el móvil no conecta

1. **Misma red WiFi:** el móvil y el portátil deben estar en la misma red. Si el portátil usa cable Ethernet y el móvil WiFi, la IP a usar es la del adaptador Ethernet del portátil, no la del WiFi.

2. **Obtener IP correcta:** en PowerShell del portátil ejecutar:
   ```powershell
   ipconfig
   ```
   Buscar **"Dirección IPv4"** bajo el adaptador activo (Wi-Fi o Ethernet).

3. **Firewall de Windows:** si el móvil no llega, el firewall puede estar bloqueando el puerto 8000.  
   Solución rápida (ejecutar como Administrador):
   ```powershell
   New-NetFirewallRule -DisplayName "PearVision QC" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
   O desactivar temporalmente el firewall privado durante las pruebas.

4. **Puerto en uso:** si el puerto 8000 ya está ocupado, usar otro:
   ```powershell
   .venv\Scripts\python.exe scripts\pearvision_qc_web_local_v1.py --port 8080
   ```

5. **Cámara no abre:** comprobar que no hay otra app (Teams, Zoom) usando la cámara. El script intenta automáticamente el índice `0` y el `1`.

---

## 7. Limitaciones

1. **El móvil NO usa su propia cámara.** Solo visualiza y controla la cámara del portátil. Para usar la cámara del móvil sería necesario una versión con WebRTC, IP Webcam app (Android) o subida de frames vía multipart POST.

2. **Latencia de vídeo:** MJPEG no es WebRTC. Hay un pequeño retardo de 0.1–0.5 s en función de la red WiFi local. Es perfectamente usable en LAN.

3. **Un solo cliente a la vez:** el generador MJPEG puede alimentar varios clientes simultáneos, pero la inferencia se ejecuta en un único hilo. El estado es el mismo para todos los clientes.

4. **Sin autenticación:** el servidor no tiene contraseña. Solo accesible en la red local; no exponerlo a internet.

5. **Iluminación y fondo:** igual que V6. Fondo blanco/gris claro, iluminación uniforme, pera completa y centrada. Sombras o fondos complejos degradan la segmentación.

6. **Solo defectos superficiales visibles.** El sistema no mide nada interno (firmeza, Brix, azúcar, etc.).

---

## 8. Salidas generadas

Las evidencias guardadas (botón o `/api/save`) se almacenan en:

```
outputs/live_camera_qc_web_v1/
  frames_original/    → frame original JPG
  frames_overlay/     → frame con contorno y etiqueta
  masks/              → máscara de segmentación
  roi_processed/      → ROI 224x224 gray_bg (entrada a U3)
  metadata/           → JSON completo por frame
  live_predictions.csv → historial CSV (mismo formato que V6)
```

---

## 9. Conclusión

`pearvision_qc_web_local_v1.py` permite acceder a PearVision QC V6 desde cualquier dispositivo en la misma red local sin ningún coste de cloud ni instalación en el móvil. El pipeline de inferencia es idéntico al de V6 (mismas constantes, mismo U3, misma política de decisión) y los datos técnicos se actualizan cada 400 ms vía JSON polling.

El script es académico, entendible y sin dependencias externas más allá de FastAPI + Uvicorn, OpenCV, PyTorch y PIL.

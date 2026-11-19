# Mine-imator Motion Capture

Aplicación de escritorio para capturar movimientos con una webcam y convertirlos en keyframes de un personaje humano de Mine-imator.

## Motor de captura 3D

**Versión 0.3.0-beta.1 — rama experimental RTMPose + MotionBERT.**

La reconstrucción utiliza el pipeline oficial `human3d` de MMPose:

- En Windows, la captura de una sola persona usa el fotograma completo como recorte (`whole_image`) y omite un detector pesado por fotograma. En otros sistemas se conserva **RTMDet-M** para localizar a la persona.
- **RTMPose-M** estima las articulaciones 2D.
- **MotionBERT** utiliza la secuencia temporal para reconstruir la pose 3D.
- El esqueleto Human3.6M de 17 articulaciones se adapta a los landmarks que consume el retargeter de Mine-imator.

El procesamiento se ejecuta en CPU para no depender de una GPU NVIDIA. La vista previa permanece dedicada al vídeo de la webcam; MotionBERT reconstruye la secuencia cuando termina REC, por lo que el procesamiento puede durar más que la grabación.

La secuencia se remuestrea a 30 FPS a partir de las marcas de tiempo reales de la cámara. Esto conserva la duración del movimiento aunque la webcam entregue una frecuencia distinta. La exportación también se genera a 30 FPS.

La aplicación inicializa los estimadores al conectar la cámara, no al terminar la primera grabación. En Windows se omite el detector de personas: el usuario debe ser la única persona visible, permanecer centrado y mantener el cuerpo completo dentro del encuadre. Esto evita ejecutar Faster R-CNN en cada fotograma. En otros sistemas se conserva RTMDet-M del alias `human3d`. Los pesos necesarios se descargan durante la primera conexión y se reutilizan en las capturas posteriores. Durante el análisis se muestra el avance, la velocidad y un tiempo restante aproximado.

Las puntuaciones de confianza se toman de las detecciones 2D de RTMPose y se convierten al orden Human3.6M antes del retargeting; así no se depende de las puntuaciones 3D de relleno del estimador. Human3.6M no incluye nariz ni orejas como articulaciones independientes. Esos puntos auxiliares se aproximan usando la cabeza, el cuello y la anchura de los hombros; no son detecciones directas.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Python 3.11 de 64 bits.
- Webcam funcional.
- Conexión a Internet durante la primera conexión de la cámara para descargar los pesos de los modelos.
- Plantilla humana `.miobject` de Mine-imator con `body`, `head`, `left_arm`, `right_arm`, `left_leg` y `right_leg`.

## Instalar desde código

En PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install numpy==1.26.4
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install mmengine==0.10.7
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
pip install six json-tricks munkres matplotlib pillow xtcocotools==1.14.3 pycocotools scipy shapely terminaltables tqdm
pip install --no-deps -r requirements-mmpose.txt
python -c "import torch, mmcv, mmengine, mmdet, mmpose; from mmpose.apis import MMPoseInferencer; print('OpenMMLab listo')"
```

## Ejecutar

```powershell
python main.py
```

## Flujo de uso

1. Conecta la webcam y espera a que la aplicación la detecte.
2. Carga tu plantilla humana `.miobject`.
3. Define la duración de la captura.
4. Colócate de cuerpo completo dentro del encuadre. Evita ocultar hombros, codos, muñecas, caderas, rodillas y tobillos.
5. Pulsa **Grabar**. Después de la cuenta regresiva, mantén una postura neutra brevemente y realiza el movimiento.
6. Al terminar REC, la aplicación prepara el vídeo temporal, normaliza sus fotogramas a 30 FPS y ejecuta RTMPose + MotionBERT.
7. Cuando aparezcan las muestras 3D, exporta el nuevo `.miobject`.

La posición global del personaje permanece ligada a la plantilla. El retargeter utiliza la primera postura corporal fiable como referencia para calcular los cambios de orientación y articulación.

## Compilación de Windows

El workflow **Compilar Windows** se ejecuta al actualizar `main` o `feature/rtmpose-motionbert`, y también puede iniciarse manualmente desde GitHub Actions. Instala las dependencias, ejecuta las pruebas, comprueba la sintaxis y compila la aplicación con PyInstaller. El resultado se publica como un único artifact descargable de Actions; no crea un GitHub Release.

## Precisión y límites

La reconstrucción 3D se infiere a partir de vídeo monocular: no equivale a medir la profundidad real con varias cámaras o sensores. La oclusión de articulaciones, la iluminación insuficiente, el desenfoque y los giros pronunciados pueden reducir la precisión. El procesamiento en CPU puede tardar, especialmente en clips largos.

## Tecnologías

- MMPose: pipeline `human3d`, RTMPose-M, RTMDet-M y MotionBERT.
- PyTorch CPU.
- OpenCV: captura de vídeo y preparación de la secuencia.
- PySide6: interfaz de escritorio.
- PyInstaller: compilación de Windows.


El modo de diagnóstico acepta `--calibration-time 0.8` para seleccionar una postura neutra de referencia. La exportación conserva el movimiento de los fotogramas anteriores y posteriores a esa postura.
# Mine-imator Motion Capture

Aplicación de escritorio para grabar los movimientos de una persona con una webcam y convertirlos en keyframes de un personaje humano de Mine-imator.

## Versión 0.2.10

- Se recupera el mapeo directo de articulaciones de la primera versión, que calcula los ángulos desde los landmarks de cada segmento sin una reconstrucción adicional de cuaterniones ni compensaciones entre jerarquías.
- Brazos: hombro–codo para la orientación del brazo y hombro–codo–muñeca para la flexión.
- Piernas: cadera–rodilla para la orientación de la pierna y cadera–rodilla–tobillo para la flexión.
- El torso y la cabeza se calculan desde sus propios landmarks.
- La pista `char` se congela en su primer transform. Se eliminan movimientos globales heredados y no se generan desplazamientos nuevos: el personaje permanece en la posición definida por la plantilla.
- La plantilla conserva campos estáticos del rig, como `POS_Z` y `TEXTURE_OBJ`, además de accesorios.
- La cola de captura mantiene los frames con timestamps durante REC y los procesa antes de habilitar la exportación.
- La cámara no se fuerza a 60 FPS; los FPS de vista previa se configuran de forma independiente.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Python 3.11 para ejecutar desde el código.
- Webcam funcional.
- Conexión a Internet la primera vez para descargar el modelo de seguimiento.
- Un archivo humano .miobject de Mine-imator que incluya `body`, `head`, `left_arm`, `right_arm`, `left_leg` y `right_leg`.

## Ejecutar desde código

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python main.py
~~~

## Flujo de uso

1. Conecta la webcam.
2. Selecciona **Heavy** para probar el modelo más exigente, **Full** para equilibrar calidad y velocidad o **Lite** si el equipo necesita más rendimiento. Ajusta los FPS de vista previa según prefieras.
3. Carga tu plantilla humana .miobject.
4. Define la duración y los FPS de exportación.
5. Colócate de cuerpo completo dentro del encuadre. Evita que hombros, codos, muñecas, caderas, rodillas y tobillos queden ocultos.
6. Pulsa **Grabar**. Se mostrará 3, 2, 1 y después comenzará la captura.
7. Al terminar, la aplicación procesa las muestras pendientes.
8. Exporta un nuevo .miobject.

El retargeting no mueve la raíz ni intenta desplazar al personaje por la escena. Cada articulación se calcula directamente a partir de la dirección de su segmento y, en codos y rodillas, de su ángulo de flexión. La posición del personaje sigue siendo la de la plantilla; cualquier animación global que ya exista en la pista `char` se reduce al primer transform.

## Compilación de Windows

El workflow **Compilar Windows** se ejecuta con cada actualización de main y también puede iniciarse manualmente desde GitHub Actions. Ejecuta pruebas y compila la aplicación con PyInstaller. El artifact de Actions contiene directamente los archivos de la aplicación; GitHub lo presenta como un solo ZIP descargable. No se crea un GitHub Release.

## Precisión y límites

MediaPipe estima coordenadas 3D desde una cámara monocular; no mide profundidad real. Los resultados pueden perder precisión si una articulación queda oculta, hay poca luz o la persona gira mucho respecto a la cámara. El retargeting conserva la correspondencia directa de los landmarks visibles; no inventa desplazamiento global ni giros de raíz para suplir datos que no se midieron.

## Tecnologías

- MediaPipe Pose Landmarker: detección corporal y landmarks 3D estimados.
- OpenCV: acceso a la webcam.
- PySide6: interfaz.
- PyInstaller: compilación de Windows.

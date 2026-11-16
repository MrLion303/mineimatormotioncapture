# Mine-imator Motion Capture

Aplicación de escritorio para grabar los movimientos de una persona con una webcam y convertirlos en keyframes de un personaje humano de Mine-imator.

## Versión 0.2.11

- El torso usa los tres canales nativos `BEND_ANGLE_X/Y/Z` de Mine-imator; la cabeza usa `ROT_X` y `ROT_Z`, siguiendo la estructura de la demostración de rig compartida.
- Brazos y piernas conservan sus canales nativos de orientación `ROT_X/Y/Z` y flexión `BEND_ANGLE_X`; la flexión del codo y la rodilla se calcula por separado.
- Cada articulación se calibra con su primera muestra válida. La postura de inicio pasa a ser el reposo de la captura, evitando exportar offsets absolutos de cámara como giros cercanos a 180°.
- La pista `char` se congela en su primer transform. No se generan desplazamientos ni rotaciones globales.
- La plantilla conserva campos estáticos del rig, como `POS_Z` y `TEXTURE_OBJ`, además de accesorios.
- Durante REC se conserva la resolución solicitada de 640×480; solo se reducen imágenes de cámaras que entreguen una resolución superior.
- La cola de captura mantiene los frames con timestamps y los procesa antes de habilitar la exportación.

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

El retargeting convierte los landmarks a los canales nativos del rig humano: flexiones XYZ para el torso, rotaciones X/Z para la cabeza y rotaciones más flexión para las extremidades. La primera muestra válida de cada articulación sirve como referencia neutra; por eso, al iniciar REC, mantén una postura natural y quieta durante un instante antes de comenzar el movimiento. La posición del personaje sigue siendo la de la plantilla y la pista `char` se reduce al primer transform.

## Compilación de Windows

El workflow **Compilar Windows** se ejecuta con cada actualización de main y también puede iniciarse manualmente desde GitHub Actions. Ejecuta pruebas y compila la aplicación con PyInstaller. El artifact de Actions contiene directamente los archivos de la aplicación; GitHub lo presenta como un solo ZIP descargable. No se crea un GitHub Release.

## Precisión y límites

MediaPipe estima coordenadas 3D desde una cámara monocular; no mide profundidad real. Los resultados pueden perder precisión si una articulación queda oculta, hay poca luz o la persona gira mucho respecto a la cámara. El retargeting conserva la correspondencia directa de los landmarks visibles; no inventa desplazamiento global ni giros de raíz para suplir datos que no se midieron.

## Tecnologías

- MediaPipe Pose Landmarker: detección corporal y landmarks 3D estimados.
- OpenCV: acceso a la webcam.
- PySide6: interfaz.
- PyInstaller: compilación de Windows.

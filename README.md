# Mine-imator Motion Capture

Aplicación de escritorio para grabar los movimientos de una persona con una webcam y convertirlos en keyframes de un personaje humano de Mine-imator.

## Qué incluye esta primera versión

- Vista previa de webcam con el esqueleto detectado superpuesto.
- Cuenta regresiva de **3 segundos** al pulsar **Grabar**.
- Duración configurable antes de grabar, de 1 a 300 segundos.
- Inicio automático al terminar la cuenta regresiva y parada automática al cumplirse la duración.
- Parada manual y cancelación de la cuenta regresiva.
- Detección corporal 3D estimada mediante MediaPipe Pose Landmarker.
- Retargeting inicial de torso, cabeza, brazos y piernas.
- Suavizado opcional e interpolación al FPS de salida.
- Reducción opcional de keyframes para evitar archivos innecesariamente densos.
- Exportación a un nuevo archivo **.miobject** a partir de una plantilla humana seleccionada.
- Conservación de los identificadores, el modelo, las partes y la jerarquía de la plantilla.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Python 3.11 para ejecutar desde el código.
- Webcam funcional.
- Conexión a Internet la primera vez para descargar el modelo Pose Landmarker.
- Un archivo humano **.miobject** de Mine-imator que incluya las pistas **body**, **head**, **left_arm**, **right_arm**, **left_leg** y **right_leg**.

## Ejecutar desde código

Abre PowerShell en la carpeta del repositorio:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
~~~

## Flujo de uso

1. Conecta la webcam y espera a que aparezca el esqueleto en la vista previa.
2. Selecciona tu archivo humano **.miobject** como plantilla.
3. Define la duración y los FPS de exportación.
4. Colócate de cuerpo completo en el encuadre. Mantén una postura inicial cómoda durante la cuenta regresiva: la primera pose detectada sirve como referencia neutra.
5. Pulsa **Grabar**. Se mostrará 3, 2, 1 y la captura comenzará al cumplirse los tres segundos.
6. La grabación terminará automáticamente al llegar a la duración elegida. También puedes detenerla antes.
7. Pulsa **Exportar .miobject** y guarda el resultado con otro nombre para no sobrescribir tu plantilla.

## Construcción para Windows

El workflow **Compilar Windows** se ejecuta en cada actualización de **main** y también puede iniciarse manualmente desde GitHub Actions. Primero ejecuta las pruebas y después compila la aplicación con PyInstaller. El ZIP se publica como artifact descargable de Actions durante 14 días; no se crea un GitHub Release.

## Estructura

- **main.py**: entrada de la aplicación.
- **mocap_app/camera.py**: webcam, modelo y seguimiento corporal.
- **mocap_app/retarget.py**: conversión de landmarks a controles del personaje.
- **mocap_app/miobject.py**: validación de plantillas y escritura de archivos Mine-imator.
- **mocap_app/main_window.py**: interfaz, cuenta regresiva y control de grabación.
- **tests/**: pruebas del retargeting y del exportador.
- **.github/workflows/compilar-windows.yml**: pruebas y compilación para Windows.

## Precisión y límites conocidos

La webcam normal no mide profundidad real. MediaPipe estima coordenadas 3D a partir de una imagen monocular; los giros hacia los lados, las articulaciones ocultas y los movimientos fuera de cámara pueden perder precisión. Esta versión genera rotaciones corporales y flexiones de codos/rodillas; todavía no reconstruye desplazamiento global real en profundidad ni torsiones finas de muñecas o manos.

El mapeo de ángulos es una primera adaptación al sistema de ejes del modelo humano de Mine-imator. Debe probarse en Mine-imator y ajustarse según el resultado visual. La aplicación conserva la estructura de la plantilla y reemplaza las pistas de animación de las partes corporales admitidas; las pistas de grupos se limpian para que una animación anterior de la plantilla no se mezcle con la captura nueva.

## Tecnología y atribución

- MediaPipe Pose Landmarker: detección de pose y estimación de landmarks.
- OpenCV: acceso a la webcam y visualización.
- PySide6: interfaz de escritorio.

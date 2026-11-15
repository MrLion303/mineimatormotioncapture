# Mine-imator Motion Capture

Aplicación de escritorio para grabar los movimientos de una persona con una webcam y convertirlos en keyframes de un personaje humano de Mine-imator.

## Mejoras de la versión 0.2.1

- Selector de cámara activo tras conectar; cambia de dispositivo y la aplicación reconecta automáticamente sin reiniciarse.

## Mejoras de la versión 0.2.0

- Cuenta regresiva de 3 segundos y duración de captura configurable.
- Selector de modelo **Full** (más detallado) o **Lite** (más ligero).
- Retargeting local relativo al torso para que inclinar o girar el cuerpo no descontrole brazos y piernas.
- Conservación de las rotaciones de reposo del archivo Mine-imator utilizado como plantilla.
- Estabilización temporal One Euro de los landmarks, con filtrado adaptativo para reducir vibraciones sin borrar movimientos lentos.
- Umbral de keyframes reducido a 0.18° y una clave de respaldo al menos cada 0.10 segundos.
- Exportación que conserva el primer transform de cada grupo como valor estático.
- El artifact de GitHub Actions se descarga como **un único ZIP**, sin crear un ZIP interno.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Python 3.11 para ejecutar desde el código.
- Webcam funcional.
- Conexión a Internet la primera vez para descargar el modelo de seguimiento.
- Un archivo humano .miobject de Mine-imator que incluya las partes body, head, left_arm, right_arm, left_leg y right_leg.

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
2. Selecciona **Full** para priorizar el detalle del seguimiento o **Lite** si el equipo necesita más rendimiento.
3. Carga tu plantilla humana .miobject.
4. Define la duración y los FPS.
5. Colócate de cuerpo completo dentro del encuadre y mantén tu postura inicial durante la cuenta regresiva.
6. Pulsa **Grabar**. Se mostrará 3, 2, 1 y después comenzará la captura.
7. La grabación termina automáticamente al llegar a la duración elegida; también puedes detenerla antes.
8. Exporta un nuevo .miobject.

La primera pose estable se utiliza como referencia neutra. La aplicación conserva los valores base de rotación de la plantilla y aplica cambios relativos respecto a esa postura, en lugar de forzar los ejes de las extremidades a cero.

## Compilación de Windows

El workflow **Compilar Windows** se ejecuta con cada actualización de main y también puede iniciarse manualmente desde GitHub Actions. Ejecuta pruebas y compila la aplicación con PyInstaller. El artifact de Actions contiene directamente los archivos de la aplicación; GitHub lo presenta como un solo ZIP descargable. No se crea un GitHub Release.

## Precisión y límites

MediaPipe estima coordenadas 3D desde una cámara monocular; no mide profundidad real. Los resultados pueden perder precisión si una articulación queda oculta, hay poca luz o la persona gira mucho respecto a la cámara. La versión 0.2.0 mejora la referencia corporal, los valores de reposo y el filtrado, pero el retargeting debe seguir validándose en Mine-imator con capturas reales.

## Tecnologías

- MediaPipe Pose Landmarker: detección corporal y landmarks 3D estimados.
- OpenCV: acceso a la webcam.
- PySide6: interfaz.
- PyInstaller: compilación de Windows.

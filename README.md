# Mine-imator Motion Capture

Aplicación de escritorio para grabar los movimientos de una persona con una webcam y convertirlos en keyframes de un personaje humano de Mine-imator.

## Versión 0.2.12

- Se fija el modelo MediaPipe Pose Landmarker Heavy, sin selector de calidad; se utilizan parámetros de confianza más estrictos.
- La vista previa usa la frecuencia real de la cámara; la exportación permite elegir 24, 30 o 60 FPS para que coincida con el tempo del proyecto de Mine-imator (24 FPS por defecto).
- El torso se reconstruye como un marco 3D ortonormal usando hombros y caderas; el giro horizontal se exporta en el canal nativo `BEND_ANGLE_Z`, conforme al rig de referencia de Mine-imator.
- Los brazos se expresan en el marco local actual del torso, evitando duplicar el giro del torso en los brazos.
- La cabeza usa `ROT_X` y `ROT_Z`; el torso usa `BEND_ANGLE_X/Y/Z` con el yaw asignado a `BEND_ANGLE_Z`.
- Brazos y piernas se orientan desde la dirección 3D del hueso proximal con dos grados de libertad observables; la flexión del codo y la rodilla se calcula por separado y no se inventa torsión axial.
- Se conserva la amplitud original de los movimientos de piernas; el ruido se controla mediante suavizado temporal, sin recortar los pasos capturados.
- El suavizado combina mediana temporal y un filtro adaptativo tipo One Euro: amortigua saltos aislados y conserva movimientos rápidos.
- El torso usa su primer marco 3D como referencia; brazos y piernas conservan su pose local desde el primer fotograma, incluso si el video comienza a mitad de un gesto. La cabeza se calibra por separado para compensar el offset anatómico de nariz y orejas.
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
2. La aplicación selecciona automáticamente el modelo **Heavy** y configura la vista previa a 30 FPS.
3. Carga tu plantilla humana .miobject.
4. Define la duración y selecciona los mismos FPS que tiene tu proyecto de Mine-imator. El valor predeterminado es 24 FPS, igual que el preset de prueba.
5. Colócate de cuerpo completo dentro del encuadre. Evita que hombros, codos, muñecas, caderas, rodillas y tobillos queden ocultos.
6. Pulsa **Grabar**. Se mostrará 3, 2, 1 y después comenzará la captura.
7. Al terminar, la aplicación procesa las muestras pendientes.
8. Exporta un nuevo .miobject.

El retargeting construye una orientación 3D del torso a partir de los ejes de hombros y caderas. Usa la primera postura corporal completa como referencia; mantén una posición natural y quieta durante un instante al comenzar REC. Los brazos se calculan en relación con el torso actual, mientras que las piernas se mantienen relacionadas con la orientación inicial de la raíz. La amplitud de las articulaciones se conserva y el filtro adaptativo reduce el ruido sin amortiguar artificialmente el gesto. La posición del personaje sigue siendo la de la plantilla y la pista `char` se reduce al primer transform.

## Compilación de Windows

El workflow **Compilar Windows** se ejecuta con cada actualización de main y también puede iniciarse manualmente desde GitHub Actions. Ejecuta pruebas y compila la aplicación con PyInstaller. El artifact de Actions contiene directamente los archivos de la aplicación; GitHub lo presenta como un solo ZIP descargable. No se crea un GitHub Release.

## Precisión y límites

MediaPipe estima coordenadas 3D desde una cámara monocular; no mide profundidad real. Los resultados pueden perder precisión si una articulación queda oculta, hay poca luz o la persona gira mucho respecto a la cámara. El retargeting conserva la correspondencia directa de los landmarks visibles; no inventa desplazamiento global ni giros de raíz para suplir datos que no se midieron.

## Tecnologías

- MediaPipe Pose Landmarker: detección corporal y landmarks 3D estimados.
- OpenCV: acceso a la webcam.
- PySide6: interfaz.
- PyInstaller: compilación de Windows.

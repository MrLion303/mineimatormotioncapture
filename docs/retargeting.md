# Retargeting de movimiento

## Datos de entrada

Cada muestra capturada contiene los landmarks 3D de MediaPipe Pose Landmarker en coordenadas de mundo y los landmarks normalizados de imagen. El retargeter utiliza las coordenadas de mundo para estimar direcciones y ángulos.

## Correspondencia de partes

| Mine-imator | Landmarks de entrada | Controles |
|---|---|---|
| body | Hombros y caderas | ROT_X, ROT_Y, ROT_Z |
| head | Nariz y orejas | ROT_X, ROT_Y, ROT_Z |
| left_arm | Hombro, codo y muñeca izquierdos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |
| right_arm | Hombro, codo y muñeca derechos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |
| left_leg | Cadera, rodilla y tobillo izquierdos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |
| right_leg | Cadera, rodilla y tobillo derechos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |

El sombrero, cuando existe en la plantilla, no recibe keyframes propios: hereda el movimiento de la cabeza.

## Referencia neutra

La primera muestra válida de cada parte se toma como referencia y sus ángulos se restan de las muestras posteriores. Conviene permanecer en una postura cómoda y visible durante el inicio. Así se evita exportar como pose permanente la orientación absoluta estimada por la cámara.

## Exportación

La secuencia se interpola al FPS elegido. El filtro opcional usa un suavizado exponencial sencillo; la optimización conserva el primer y último fotograma y añade keyframes cuando hay cambios angulares relevantes o se alcanza el intervalo máximo entre claves.

El exportador parte de una plantilla real de Mine-imator. Comprueba que sea un modelo humano y que tenga las seis pistas requeridas. Conserva el resto de la estructura JSON, los IDs, los metadatos y las propiedades de la plantilla. Las pistas de carpetas se vacían y los keyframes de las seis partes se sustituyen por la nueva captura.

## Limitaciones

La coordenada Z de una sola webcam es una estimación. La rotación de muñecas, dedos, inclinación fina de la cabeza y desplazamiento global en profundidad no se reconstruyen con precisión en esta versión. El resultado debe revisarse en Mine-imator; la correspondencia de ejes puede necesitar calibración adicional para distintos modelos o plantillas.

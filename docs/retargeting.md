# Retargeting de movimiento

## Qué se lee de la cámara

Cada muestra contiene landmarks 3D estimados por MediaPipe, landmarks 2D normalizados de la imagen, el tamaño de la imagen y el instante monotónico en que se capturó el frame. La exportación respeta esos timestamps y remuestrea la trayectoria al FPS de Mine-imator elegido.

La estabilización, cuando está activa, filtra una sola vez los landmarks 2D y 3D con One Euro Filter. No se vuelve a filtrar la curva de ángulos después del retargeting, para no añadir un segundo retraso ni atenuar gestos rápidos.

## Jerarquía del rig

La jerarquía se obtiene de los identificadores y de `parent` dentro del `.miobject`:

| Pista de Mine-imator | Landmarks de referencia | Qué se anima |
|---|---|---|
| `char` | Centro de caderas y línea de caderas | Posición global y orientación |
| `body` | Hombros y caderas | Inclinación, giro y torsión del torso |
| `head` | Nariz y laterales de la cabeza | Giro local de cabeza |
| `left_arm` / `right_arm` | Hombro, codo y muñeca | Orientación 3D y flexión de codo |
| `left_leg` / `right_leg` | Cadera, rodilla y tobillo | Orientación 3D y flexión de rodilla |

Los rigs incluidos como referencia tienen el torso como hijo de `char`, cabeza y brazos como hijos del torso y piernas como hijos de `char`. El exportador conserva los valores estáticos y los ángulos de reposo que encuentra en la plantilla.

## Orientación de raíz y piernas

La orientación global de pelvis se calcula únicamente desde la línea que une las dos caderas. Las rodillas y los tobillos no intervienen en ese cálculo porque se mueven de forma alternada durante una caminata. Así se evita convertir el ciclo de pasos en un giro ficticio de todo el personaje.

Las piernas se resuelven después en el marco local de la raíz. Brazos y cabeza se resuelven en el marco del torso cuando la jerarquía los hace depender de él.

## Orientación de las extremidades

Para cada brazo o pierna se usan los dos segmentos de la articulación:
- Brazo: hombro a codo y codo a muñeca.
- Pierna: cadera a rodilla y rodilla a tobillo.

Con ambos vectores se construye una base local a partir del eje del segmento superior y la normal del plano de flexión. De este modo, la orientación conserva la dirección y la torsión de la extremidad, mientras que la magnitud del ángulo del codo o la rodilla se exporta una sola vez en `BEND_ANGLE_X`, el canal utilizado por el rig de caminata proporcionado. El cambio de orientación se compone como cuaternión local con la rotación de reposo de la plantilla; no se suman componentes Euler por separado.

Cuando la extremidad está casi recta, su plano de flexión deja de estar bien definido. En ese caso se conserva la referencia anterior para reducir inversiones repentinas de 180 grados.

## Desplazamiento global

Los landmarks 3D de MediaPipe están centrados en las caderas, por lo que no bastan por sí solos para medir el desplazamiento completo de la persona por la habitación. La traslación se estima combinando:
- El centro de caderas 2D de la imagen para el movimiento horizontal y vertical en el encuadre.
- La altura del cuerpo estimada en world landmarks frente a su tamaño en píxeles para aproximar la escala y el cambio relativo de profundidad.

La traslación se escribe en `char.POS_X`, `char.POS_Y` y `char.POS_Z`. El valor inicial procede de `default_values` o del primer keyframe de la raíz. El control **Escala de desplazamiento** ajusta las unidades de Mine-imator por metro estimado. La profundidad utiliza un campo de visión vertical aproximado; por eso debe calibrarse con la cámara y no representa una medición óptica exacta.

Si los landmarks necesarios para medir la posición quedan ocultos durante unos fotogramas, se conserva la última posición válida para evitar que el personaje salte al punto de inicio.

## Interpolación y keyframes

Durante REC cada frame de cámara se guarda en una cola cronológica como JPEG de alta calidad, junto con su timestamp monotónico. La inferencia procesa todos los frames pendientes después de detener la toma; la exportación no se habilita hasta que la cola se vacía. La cola admite hasta 18.000 frames (cinco minutos a 60 FPS).

Las muestras se ordenan por tiempo de captura y se interpolan al FPS de exportación. La opción de optimización conserva el primer y último fotograma, introduce claves por cambio angular y limita el intervalo máximo entre claves. Desactivarla escribe todos los fotogramas.

El filtro de landmarks es la única etapa de estabilización; no se aplica otro suavizado recursivo sobre las rotaciones exportadas.

## Límites de una sola cámara

Una webcam RGB no mide profundidad absoluta, no observa las articulaciones que quedan detrás del cuerpo y no puede reconstruir con exactitud los grados de libertad de muñecas o dedos que no forman parte del esqueleto Pose de 33 landmarks. La traslación lateral y vertical es relativa al encuadre; la profundidad es una estimación basada en la escala aparente. Oclusiones, ropa holgada, perspectiva, iluminación, movimiento rápido y límites del modelo pueden producir errores antes de que el retargeting reciba los datos.

Por esa razón, el objetivo del retargeter es conservar fielmente las articulaciones visibles y evitar errores de jerarquía o exportación; no puede garantizar identidad geométrica perfecta entre una captura monocular y un rig 3D.

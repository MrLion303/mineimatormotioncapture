# Retargeting de movimiento (0.2.12)

## Principio de conversión

El conversor utiliza los canales nativos del rig humano de Mine-imator y calibra la postura inicial. Los landmarks de MediaPipe están expresados en el sistema de coordenadas de la cámara; no deben escribirse como rotaciones absolutas del personaje. En lugar de calcular tres ángulos independientes desde una sola línea, se construye un marco 3D ortonormal con los centros de hombros y caderas, el eje de hombros y su perpendicular. La rotación que se exporta es la diferencia entre el marco inicial y el marco actual, descompuesta en XYZ. Esto permite medir por separado una reverencia, una inclinación lateral y un giro del torso.

La raíz `char` permanece fija. No se generan traslaciones, giros globales ni desplazamientos de las articulaciones.

## Correspondencia de landmarks y canales

| Pista de Mine-imator | Landmarks | Canales animados |
|---|---|---|
| `body` | Eje entre hombros y caderas, orientación de hombros | `BEND_ANGLE_X`, `BEND_ANGLE_Y`, `BEND_ANGLE_Z` |
| `head` | Nariz y eje entre orejas | `ROT_X`, `ROT_Z` |
| `left_arm` | Hombro 11, codo 13, muñeca 15 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |
| `right_arm` | Hombro 12, codo 14, muñeca 16 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |
| `left_leg` | Cadera 23, rodilla 25, tobillo 27 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |
| `right_leg` | Cadera 24, rodilla 26, tobillo 28 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |

El torso usa los canales nativos `BEND_ANGLE_X/Y/Z` del rig humano de Mine-imator. La correspondencia se adapta al eje local de la pieza: la inclinación hacia delante/atrás se exporta en `BEND_ANGLE_X`, la inclinación lateral en `BEND_ANGLE_Y` y el giro horizontal (yaw) en `BEND_ANGLE_Z`. Esta correspondencia coincide con el archivo de referencia proporcionado, donde el giro correcto del torso está animado en la pista `body` mediante `BEND_ANGLE_Z`. La cabeza y los brazos son hijos de `body`, por lo que heredan ese giro mediante la jerarquía del rig. La cabeza utiliza X y Z relativos al marco del torso. Los brazos se miden en el marco local del torso actual, de modo que el giro del torso no se duplica como giro del brazo. Las piernas se miden respecto al marco de referencia fijo porque son hijas de `char`, no del torso. Brazos y piernas usan orientación y flexión por separado: el vector del segmento superior produce las rotaciones y el ángulo de tres puntos produce la flexión del codo o la rodilla. No se inventa torsión axial, por lo que `ROT_Y` de las extremidades permanece en cero.

## Calibración de postura inicial

El torso toma como referencia su primer marco 3D completo para medir el movimiento de giro e inclinación sin animar la raíz. La cabeza conserva su primera orientación fiable como línea base porque la distancia anatómica entre nariz y orejas no representa una inclinación del cuello. Los brazos y las piernas no se ponen a cero en el primer fotograma: sus ángulos se calculan respecto a los ejes de reposo del rig, por lo que una grabación que empieza con un brazo levantado o una rodilla flexionada conserva esa pose desde el fotograma cero.

Para estabilizar la orientación del torso y la cabeza, conviene comenzar REC con el cuerpo erguido y la cabeza mirando al frente durante un instante. No es necesario bajar los brazos ni estirar las piernas si el movimiento ya comenzó: esa postura inicial se conserva en la animación.

## Raíz y posición de escena

La pista `char` se congela en el primer transform que encuentre. Si la plantilla tenía keyframes de posición o rotación global, se conservan únicamente los valores del primer fotograma; si no tenía keyframes, se mantienen sus `default_values`. No se exporta movimiento global a `char`.

## Conservación de la plantilla

El exportador conserva las propiedades estáticas que no anima, incluidas las posiciones de partes como `POS_Z`, referencias de textura como `TEXTURE_OBJ` y pistas accesorias. La jerarquía existente en el `.miobject` no se reescribe. Los canales nativos que sí se animan se actualizan en las pistas correspondientes.

## Keyframes y suavizado

Las muestras se ordenan por su timestamp y se interpolan a 30 FPS. La optimización retiene el primer y último fotograma, agrega claves al superar aproximadamente 0.8° de cambio y limita el espacio entre claves a 0.15 segundos.

La estabilización combina una mediana temporal de tres muestras para rechazar saltos aislados y un filtro adaptativo tipo One Euro. Durante gestos rápidos aumenta el corte del filtro para no aplanar movimientos intencionales. Las piernas reciben una ganancia de 0.72 en rotación y 0.68 en flexión, además de una zona muerta de 1–1.5° para limitar la exageración de pequeñas variaciones del tracking.

## Captura de cámara

El modelo Heavy es el único modelo disponible en la interfaz. Se utilizan umbrales de confianza 0.55 para detección/presencia y 0.60 para tracking. La vista previa y la exportación están fijadas a 30 FPS.

Durante REC cada frame se guarda en una cola cronológica como JPEG de alta calidad, junto con su timestamp monotónico. La captura conserva la resolución solicitada de 640×480; únicamente reduce imágenes mayores a ese tamaño. Esto evita que la inferencia durante REC trabaje con una imagen reducida respecto al seguimiento de vista previa. La inferencia procesa los frames pendientes después de detener la toma; la exportación se habilita cuando la cola se vacía.

## Límites de una sola cámara

Una webcam RGB no mide profundidad absoluta, no observa articulaciones que quedan detrás del cuerpo y no reconstruye los grados de libertad de muñecas o dedos que no forman parte del esqueleto Pose de 33 landmarks. Oclusiones, ropa holgada, perspectiva, iluminación y movimiento rápido pueden producir errores antes de que el retargeting reciba los datos. El conversor evita añadir un desplazamiento global o rotación de raíz cuando esos movimientos no forman parte de lo que se quiere exportar.

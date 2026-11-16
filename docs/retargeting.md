# Retargeting de movimiento

## Principio de conversión

El retargeting vuelve al mapeo directo de la primera versión. Cada pista de Mine-imator recibe ángulos calculados desde sus propios landmarks de MediaPipe. No se reconstruye una jerarquía 3D completa mediante cuaterniones, no se utiliza una pose neutra para restar orientaciones y no se compensa el movimiento de una articulación contra el de su padre.

El objetivo es que el movimiento visible en el esqueleto de cámara llegue a los canales de articulación correspondientes, sin añadir traslación o giro global del personaje.

## Correspondencia de landmarks

| Pista de Mine-imator | Landmarks | Canales animados |
|---|---|---|
| `body` | Hombros y caderas | `ROT_X`, `ROT_Y`, `ROT_Z` |
| `head` | Nariz y laterales de la cabeza | `ROT_X`, `ROT_Y`, `ROT_Z` |
| `left_arm` | Hombro 11, codo 13, muñeca 15 | `ROT_X`, `ROT_Z`, `BEND_ANGLE_X` |
| `right_arm` | Hombro 12, codo 14, muñeca 16 | `ROT_X`, `ROT_Z`, `BEND_ANGLE_X` |
| `left_leg` | Cadera 23, rodilla 25, tobillo 27 | `ROT_X`, `ROT_Z`, `BEND_ANGLE_X` |
| `right_leg` | Cadera 24, rodilla 26, tobillo 28 | `ROT_X`, `ROT_Z`, `BEND_ANGLE_X` |

## Orientación de brazos y piernas

Para cada segmento principal se obtiene el vector desde la articulación de origen hasta la siguiente articulación. La inclinación se calcula directamente:

`ROT_X = atan2(\Delta z, \Delta y \cdot downSign)`

`ROT_Z = atan2(\Delta x, \Delta y \cdot downSign)`

El signo vertical `downSign` se infiere de la dirección de las espinillas visibles, para admitir ambas convenciones del eje Y de MediaPipe. El mapeo directo usa `ROT_Y = 0` en los miembros; no intenta inventar torsión axial que el esqueleto de 33 landmarks no entrega de forma fiable.

La flexión se calcula por separado con los tres puntos de cada extremidad. Se obtiene el ángulo interior entre los dos segmentos y se convierte en flexión como `180° - ánguloInterior`, limitada al rango admitido por el retargeter. Así, la orientación del brazo o pierna y la flexión del codo o rodilla son magnitudes separadas.

## Torso y cabeza

El torso se orienta a partir del eje entre los puntos medios de hombros y caderas, y del eje entre ambos hombros. La cabeza se orienta a partir de la nariz y del eje entre orejas. Esas pistas siguen correspondiendo al movimiento capturado del torso y la cabeza, pero no modifican la pista global del personaje.

## Raíz y posición de escena

La pista `char` se congela en el primer transform que encuentre. Si la plantilla tenía keyframes de posición o rotación global, se conservan únicamente los valores del primer fotograma; si no tenía keyframes, se mantiene su `default_values`. No se exporta movimiento global de cámara a `char`.

## Conservación de la plantilla

El exportador conserva propiedades que no anima, incluidas las posiciones de partes como `POS_Z`, referencias de textura como `TEXTURE_OBJ`, y pistas accesorias. La jerarquía existente en el `.miobject` no se reescribe.

## Keyframes y suavizado

Las muestras se ordenan por su timestamp y se interpolan al FPS de exportación. La optimización retiene el primer y último fotograma, agrega claves al superar aproximadamente 1° de cambio y limita el espacio entre claves a 0.20 segundos. Desactivar la optimización escribe todos los fotogramas.

La opción de estabilización aplica el suavizado de ángulos del exportador original. Se puede desactivar para conservar más literalmente las variaciones detectadas.

## Captura de cámara

Durante REC cada frame se guarda en una cola cronológica como JPEG de alta calidad, junto con su timestamp monotónico. La inferencia procesa los frames pendientes después de detener la toma; la exportación se habilita cuando la cola se vacía. La cola admite hasta 18.000 frames, equivalente a cinco minutos a 60 FPS.

## Límites de una sola cámara

Una webcam RGB no mide profundidad absoluta, no observa articulaciones que quedan detrás del cuerpo y no reconstruye los grados de libertad de muñecas o dedos que no forman parte del esqueleto Pose de 33 landmarks. Oclusiones, ropa holgada, perspectiva, iluminación y movimiento rápido pueden producir errores antes de que el retargeting reciba los datos. El conversor evita añadir un desplazamiento global o rotación de raíz cuando esos movimientos no forman parte de lo que el usuario quiere exportar.

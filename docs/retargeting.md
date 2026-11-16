# Retargeting de movimiento

## Principio de conversión

El conversor utiliza los canales nativos del rig humano de Mine-imator y calibra la postura inicial de cada articulación. Los landmarks de MediaPipe están expresados en el sistema de coordenadas de la cámara; no deben escribirse como rotaciones absolutas del personaje. En la exportación, el primer valor fiable de cada canal se toma como referencia y los keyframes contienen el cambio respecto a esa postura.

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

El torso se anima con los tres canales `BEND_ANGLE`, no con `ROT_X/Y/Z`. La cabeza utiliza X y Z; no se escribe una rotación Y que no aparece en la referencia de rig compartida. Brazos y piernas usan orientación y flexión por separado: el vector del segmento superior produce las rotaciones y el ángulo de tres puntos produce la flexión del codo o la rodilla. No se inventa torsión axial, por lo que `ROT_Y` de las extremidades permanece en cero.

## Calibración de postura inicial

Cada pista y canal se calibra con su primera muestra válida. Esa medición se resta de las muestras siguientes usando la diferencia angular más corta, evitando saltos de 360°. Así, los offsets de la cámara (por ejemplo, valores iniciales cercanos a 170°) no se convierten en una rotación enorme del personaje. Una articulación que aparece más tarde se calibra al hacerse visible por primera vez.

Al comenzar REC, la persona debe permanecer un instante en postura neutra y después ejecutar el movimiento. Esta referencia inicial es esencial: la aplicación no puede inferir cuál sería la postura de reposo si la grabación comienza mientras ya se está realizando un gesto.

## Raíz y posición de escena

La pista `char` se congela en el primer transform que encuentre. Si la plantilla tenía keyframes de posición o rotación global, se conservan únicamente los valores del primer fotograma; si no tenía keyframes, se mantienen sus `default_values`. No se exporta movimiento global a `char`.

## Conservación de la plantilla

El exportador conserva las propiedades estáticas que no anima, incluidas las posiciones de partes como `POS_Z`, referencias de textura como `TEXTURE_OBJ` y pistas accesorias. La jerarquía existente en el `.miobject` no se reescribe. Los canales nativos que sí se animan se actualizan en las pistas correspondientes.

## Keyframes y suavizado

Las muestras se ordenan por su timestamp y se interpolan al FPS de exportación. La optimización retiene el primer y último fotograma, agrega claves al superar aproximadamente 1° de cambio y limita el espacio entre claves a 0.20 segundos. Desactivar la optimización escribe todos los fotogramas.

La opción de estabilización aplica un suavizado ligero a los canales angulares. Se puede desactivar para conservar más literalmente las variaciones detectadas.

## Captura de cámara

Durante REC cada frame se guarda en una cola cronológica como JPEG de alta calidad, junto con su timestamp monotónico. La captura conserva la resolución solicitada de 640×480; únicamente reduce imágenes mayores a ese tamaño. Esto evita que la inferencia durante REC trabaje con una imagen reducida respecto al seguimiento de vista previa. La inferencia procesa los frames pendientes después de detener la toma; la exportación se habilita cuando la cola se vacía.

## Límites de una sola cámara

Una webcam RGB no mide profundidad absoluta, no observa articulaciones que quedan detrás del cuerpo y no reconstruye los grados de libertad de muñecas o dedos que no forman parte del esqueleto Pose de 33 landmarks. Oclusiones, ropa holgada, perspectiva, iluminación y movimiento rápido pueden producir errores antes de que el retargeting reciba los datos. El conversor evita añadir un desplazamiento global o rotación de raíz cuando esos movimientos no forman parte de lo que se quiere exportar.

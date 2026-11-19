# Retargeting de movimiento (0.2.12)

## Principio de conversión

El conversor utiliza los canales nativos del rig humano de Mine-imator y calibra la postura inicial. Los landmarks de MediaPipe están expresados en el sistema de coordenadas de la cámara; no deben escribirse como rotaciones absolutas del personaje. En lugar de calcular tres ángulos independientes desde una sola línea, se construye un marco 3D ortonormal con los centros de hombros y caderas, el eje de hombros y su perpendicular. La rotación que se exporta es la diferencia entre el marco inicial y el marco actual, descompuesta en XYZ. Esto permite medir por separado una reverencia, una inclinación lateral y un giro del torso.

La raíz `char` permanece fija. No se generan traslaciones, giros globales ni desplazamientos de las articulaciones.

## Correspondencia de landmarks y canales

| Pista de Mine-imator | Landmarks | Canales animados |
|---|---|---|
| `body` | Eje entre hombros y caderas, orientación de hombros | `ROT_X`, `ROT_Y`, `ROT_Z` |
| `head` | Eje cuello-cabeza (MotionBERT) o nariz y orejas reales si la fuente los detecta | `ROT_X`, `ROT_Z` |
| `left_arm` | Hombro 11, codo 13, muñeca 15 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |
| `right_arm` | Hombro 12, codo 14, muñeca 16 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |
| `left_leg` | Cadera 23, rodilla 25, tobillo 27 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |
| `right_leg` | Cadera 24, rodilla 26, tobillo 28 | `ROT_X`, `ROT_Y`, `ROT_Z`, `BEND_ANGLE_X` |

El torso se anima con los tres canales `ROT_X/Y/Z`; los valores `BEND_ANGLE` estáticos del preset se conservan sin modificarse. La cabeza utiliza X y Z. Los brazos se miden en el marco local del torso actual, de modo que el giro del torso no se duplica como giro del brazo. Las piernas se miden respecto al marco de referencia fijo porque son hijas de `char`, no del torso. Brazos y piernas usan orientación y flexión por separado: el vector del segmento superior produce las rotaciones y el ángulo de tres puntos produce la flexión del codo o la rodilla. No se inventa torsión axial, por lo que `ROT_Y` de las extremidades permanece en cero.

## Calibración de postura inicial

Cada pista y canal se calibra con su primera muestra válida. Esa medición se resta de las muestras siguientes usando la diferencia angular más corta, evitando saltos de 360°. Así, los offsets de la cámara (por ejemplo, valores iniciales cercanos a 170°) no se convierten en una rotación enorme del personaje. Una articulación que aparece más tarde se calibra al hacerse visible por primera vez.

Al comenzar REC, la persona debe permanecer un instante en postura neutra y después ejecutar el movimiento. Esta referencia inicial es esencial: la aplicación no puede inferir cuál sería la postura de reposo si la grabación comienza mientras ya se está realizando un gesto.

## Raíz y posición de escena

La pista `char` se congela en el primer transform que encuentre. Si la plantilla tenía keyframes de posición o rotación global, se conservan únicamente los valores del primer fotograma; si no tenía keyframes, se mantienen sus `default_values`. No se exporta movimiento global a `char`.

## Conservación de la plantilla

El exportador conserva las propiedades estáticas que no anima, incluidas las posiciones de partes como `POS_Z`, referencias de textura como `TEXTURE_OBJ` y pistas accesorias. La jerarquía existente en el `.miobject` no se reescribe. Los canales nativos que sí se animan se actualizan en las pistas correspondientes.

## Keyframes y suavizado

Las muestras se ordenan por su timestamp y se interpolan a 30 FPS. La optimización retiene el primer y último fotograma, agrega claves al superar aproximadamente 0.8° de cambio y limita el espacio entre claves a 0.15 segundos.

La estabilización combina una mediana temporal de tres muestras para rechazar saltos aislados y un filtro adaptativo tipo One Euro. Durante gestos rápidos aumenta el corte del filtro para no aplanar movimientos intencionales. Las piernas conservan una ganancia de movimiento de 1.0 y aplican una zona muerta de 1° en rotación y 1.5° en flexión; así se evitan pequeñas vibraciones sin comprimir la amplitud de los gestos. La exportación también mantiene el último signo fiable de la flexión cuando la geometría 3D no permite distinguir hacia qué lado se dobla la rodilla.

## Captura de cámara

El modelo Heavy es el único modelo disponible en la interfaz. Se utilizan umbrales de confianza 0.55 para detección/presencia y 0.60 para tracking. La vista previa y la exportación están fijadas a 30 FPS.

Durante REC cada frame se guarda en una cola cronológica como JPEG de alta calidad, junto con su timestamp monotónico. La captura conserva la resolución solicitada de 640×480; únicamente reduce imágenes mayores a ese tamaño. Esto evita que la inferencia durante REC trabaje con una imagen reducida respecto al seguimiento de vista previa. La inferencia procesa los frames pendientes después de detener la toma; la exportación se habilita cuando la cola se vacía.

## Límites de una sola cámara

Una webcam RGB no mide profundidad absoluta, no observa articulaciones que quedan detrás del cuerpo y no reconstruye los grados de libertad de muñecas o dedos que no forman parte del esqueleto Pose de 33 landmarks. Oclusiones, ropa holgada, perspectiva, iluminación y movimiento rápido pueden producir errores antes de que el retargeting reciba los datos. El conversor evita añadir un desplazamiento global o rotación de raíz cuando esos movimientos no forman parte de lo que se quiere exportar.

## Seguimiento de cabeza con MotionBERT

Human3.6M no proporciona nariz ni orejas. Por ello, la salida de MotionBERT conserva el eje real cuello-cabeza y no inventa puntos faciales a partir de los hombros. La orientación relativa del eje se calcula dentro del marco del torso. Este es un estimador de inclinación, no de giro axial de la cabeza; el giro `ROT_Y` no se inventa cuando el esqueleto no contiene información suficiente. Si la fuente sí entrega nariz y orejas reales (por ejemplo, una pose facial), se mantiene disponible esa ruta de retargeting.

## Continuidad de flexión

La flexión de rodilla se determina por la geometría 3D de cadera, rodilla y tobillo. Cuando la componente que distingue flexión hacia delante y hacia atrás representa menos del 12% del eje de flexión, la dirección es ambigua; en ese caso se conserva el último signo fiable para impedir inversiones de 180° causadas por pequeñas variaciones de profundidad. Cuando la geometría vuelve a ser concluyente, vuelve a prevalecer la medición actual.

## Diagnóstico visual

Además del archivo `.poses.json` con landmarks 3D, el test de regresión guarda las detecciones 2D COCO de RTMPose sobre el vídeo original en `*.seguimiento.mp4`. El informe registra cobertura por parte, uso del respaldo de torso, máximos de variación por canal y saltos superiores a 60°. El workflow también compone un `.miproject` temporal que conserva los ajustes de escena/render del preset original y sustituye únicamente sus timelines por los de la animación exportada, para abrirlo con Mine-imator Community Build 1.0.0.

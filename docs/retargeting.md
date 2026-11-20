# Retargeting de movimiento (0.2.13)

## Principio de conversión

El conversor usa MediaPipe Pose Heavy para recuperar 33 landmarks por fotograma y los convierte al sistema de coordenadas del rig humano de Mine-imator. Los landmarks 3D de MediaPipe están centrados en la pelvis: describen la postura relativa, pero no proporcionan una traslación global fiable del personaje. Por eso la orientación se obtiene desde el marco del torso y la traslación de escena se estima por separado usando la imagen y el pie que permanece más bajo.

Las extremidades ya no se exportan como una aproximación formada por la rotación del hueso superior más un ángulo de flexión sin signo. Se exportan mediante la cinemática inversa (IK) nativa de Mine-imator: cada brazo y pierna tiene un objeto objetivo para la muñeca o el tobillo y un objeto polo para el codo o la rodilla. Mine-imator resuelve los dos segmentos del miembro con su rig real.

## Correspondencia de pistas y landmarks

| Pista de Mine-imator | Landmarks | Tratamiento |
|---|---|---|
| `body` | Hombros 11/12 y caderas 23/24 | Orientación relativa en `BEND_ANGLE_X/Y/Z` |
| `head` | Nariz 0 y orejas 7/8 | Giro relativo a torso en `ROT_X/Y/Z` |
| `left_arm` | Hombro 11, codo 13, muñeca 15 | IK a muñeca + polo de codo |
| `right_arm` | Hombro 12, codo 14, muñeca 16 | IK a muñeca + polo de codo |
| `left_leg` | Cadera 23, rodilla 25, tobillo 27 | IK a tobillo + polo de rodilla |
| `right_leg` | Cadera 24, rodilla 26, tobillo 28 | IK a tobillo + polo de rodilla |

El torso usa un marco 3D ortonormal construido con hombros y caderas. La diferencia entre el marco inicial y el actual se descompone en flexión frontal, inclinación lateral y giro horizontal. La cabeza y los brazos heredan la orientación del torso; las piernas son hijas directas del personaje y no heredan el giro del torso.

## Cinemática inversa y dimensiones del rig

Los objetivos IK se crean como ocho pistas auxiliares de tipo cámara, ocultas y bloqueadas, hijas de la pista principal `char`. No contienen geometría visible. Cuatro pistas representan objetivos de muñeca/tobillo y otras cuatro representan polos de codo/rodilla. Las pistas de los miembros apuntan a esos objetos mediante `IK_TARGET`, `IK_TARGET_ANGLE` y `IK_BLEND=1`.

La plantilla Steve de Mine-imator usa estas longitudes del rig: brazos de 4 unidades en el segmento superior y 6 en el inferior; piernas de 6 y 6. La captura determina las direcciones de ambos segmentos y el exportador las adapta a esas longitudes nativas. La posición del polo se calcula desde el codo o la rodilla capturados, por lo que la flexión conserva el lado correcto de la articulación.

Para evitar doble transformación, los canales manuales de giro/flexión de brazos y piernas se dejan en cero cuando IK está activa. El motor IK es quien resuelve la orientación y el ángulo de la articulación a partir del objetivo y el polo. Los canales del cuerpo y de la cabeza siguen siendo animados directamente.

## Geometría de piernas y apoyo del suelo

Para la dirección y flexión visible de las piernas, la imagen 2D es la señal principal. La profundidad estimada de los landmarks 3D solo aporta una contribución pequeña y limitada, porque la profundidad monocular puede hacer que una pierna visualmente recta parezca doblada. La cadera, la rodilla y el tobillo se calculan por separado; no se fuerza una flexión constante ni se aplana la pose cuando una pierna se levanta.

La posición de la raíz se calcula con las caderas en coordenadas de imagen para el movimiento horizontal y con el tobillo más bajo para mantener el apoyo. La escala usa las 32 unidades de modelo del Steve estándar (dos bloques de 16 unidades); usar la altura métrica de 1.8 metros como si fueran unidades de Mine-imator reduciría la corrección de raíz casi 18 veces y dejaría los pies separados del suelo. La traslación global se exporta en la pista `char`; los objetivos y polos IK, como hijos de `char`, acompañan esa traslación. La profundidad global absoluta no se infiere de una webcam RGB, así que el desplazamiento de raíz en Z permanece en cero en la ruta 2D.

## Calibración y tiempo

La primera pose completa del torso define el marco de referencia. La pose inicial de brazos y piernas no se usa como una postura neutral ficticia: las direcciones de los miembros se convierten directamente en objetivos IK, por lo que un brazo que empieza levantado conserva esa posición en el primer fotograma.

El vídeo original se analiza a su frecuencia de origen y las muestras se remuestrean a los FPS elegidos para el proyecto. Los objetivos IK usan una separación máxima de clave más corta que los canales angulares para conservar gestos rápidos. La suavización temporal se aplica a los objetivos y polos por separado, sin reemplazar la trayectoria por ángulos promedio.

La regresión de Rickroll valida el vídeo original completo (1,807 fotogramas, 60 FPS) y exporta el proyecto de referencia a 24 FPS para coincidir con el tempo de la plantilla de Mine-imator Community Build 1.0.0.

## Límites de una sola cámara

Una webcam RGB no mide profundidad absoluta ni puede recuperar con certeza una articulación completamente ocluida, la torsión axial de muñecas o detalles de dedos. En esos casos el objetivo IK es una estimación a partir de la proyección 2D y de la profundidad relativa. Las oclusiones, la perspectiva, la ropa holgada y los movimientos muy rápidos pueden introducir errores antes del retargeting.

La nueva salida mejora la correspondencia de manos, pies y bisagras al utilizar los objetivos y polos reales del rig, pero una compilación o una regresión matemática no certifican por sí solas que la animación se vea idéntica. La comprobación visual final requiere abrir el `.miobject` en Mine-imator y revisar el movimiento en el visor.

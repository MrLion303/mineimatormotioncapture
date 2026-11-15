# Retargeting de movimiento

La versión 0.2.0 calcula las rotaciones de las partes del personaje respecto a una postura de referencia y al marco local del torso.

## Marco corporal

Se construye un sistema ortonormal con:
- Eje X: anchura estimada del cuerpo, promediando hombros y caderas.
- Eje Y: dirección entre el punto medio de las caderas y el de los hombros, ortogonalizada respecto a X.
- Eje Z: producto cruzado entre X e Y.

Los vectores de brazos y piernas se expresan en este marco local. Así, una rotación del cuerpo no se interpreta automáticamente como una rotación independiente de cada extremidad.

## Correspondencia de partes

| Mine-imator | Landmarks | Controles |
|---|---|---|
| body | Hombros y caderas | ROT_X, ROT_Y, ROT_Z |
| head | Nariz, orejas u ojos | ROT_X, ROT_Y, ROT_Z |
| left_arm | Hombro, codo y muñeca izquierdos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |
| right_arm | Hombro, codo y muñeca derechos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |
| left_leg | Cadera, rodilla y tobillo izquierdos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |
| right_leg | Cadera, rodilla y tobillo derechos | ROT_X, ROT_Y, ROT_Z, BEND_ANGLE_X |

La orientación del brazo o pierna se obtiene de la rotación entre el vector de la postura de referencia y el vector actual, ambos expresados en el marco local del torso. La flexión se calcula por separado a partir del ángulo entre los dos segmentos de la extremidad.

## Postura de referencia y valores base

El primer fotograma válido con el cuerpo suficientemente visible se utiliza como postura neutra. El exportador lee los valores del primer keyframe de cada parte en el archivo humano de Mine-imator y aplica los cambios de movimiento sobre esa base.

Esto es importante para los brazos: las plantillas humanas suelen tener ángulos de reposo diferentes a izquierda y derecha. No se sustituyen por ROT_X/Y/Z = 0.

Los grupos se conservan con el transform del primer keyframe, convertido en un valor estático en el fotograma cero. De ese modo, no se arrastra al resultado la animación antigua de la plantilla, pero se mantiene su colocación.

## Estabilización y keyframes

Antes del retargeting, los landmarks 3D pasan por filtros One Euro dependientes del tiempo. El filtro atenúa vibraciones pequeñas y adapta su respuesta cuando el movimiento se acelera.

La secuencia se interpola al FPS seleccionado. La optimización conserva el primer y último fotograma y escribe una clave cuando el cambio angular acumulado llega a 0.18 grados o cuando han pasado 0.10 segundos desde la última clave.

## Limitaciones

La coordenada 3D estimada por una sola webcam no equivale a una medición óptica de profundidad. Las oclusiones, la baja iluminación y las rotaciones fuertes respecto a la cámara pueden generar landmarks incorrectos. La rotación de muñecas y dedos, el desplazamiento real en profundidad y las torsiones finas todavía no se reconstruyen de forma completa.

Los ángulos se convierten al sistema de ejes de Mine-imator mediante una correspondencia inicial. El resultado debe validarse visualmente con la plantilla que se vaya a utilizar.

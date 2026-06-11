# Roadmap para demostrar comunicación emergente

## Estado actual

- La política ya resuelve bien el obstáculo simple y la U.
- El desempeño es estable incluso sin BG (`--no_bg`).
- Los agentes parecen llegar casi al mismo tiempo, así que la vocalización actual probablemente no es causal para el éxito.
- Conclusión: ya hay buena navegación, pero todavía no hay evidencia fuerte de comunicación emergente funcional.

## Qué se cambió en el canal social

Antes, el canal social era esencialmente:
- ángulo relativo al otro agente.

Ahora, el canal social es:
- ángulo relativo al otro agente multiplicado por una intensidad que decae con la distancia.

En forma conceptual:

`social_signal = rel_angle_social * signal_strength`

con:

`signal_strength = exp(-dist / 180.0)`

Esto mantiene un solo canal social (no cambia la dimensión de entrada de la red), pero hace que la señal sea:
- más fuerte cuando el otro agente está cerca,
- más débil cuando está lejos,
- y con signo según la dirección relativa.

## Objetivo siguiente

Pasar de:
- “la red navega bien”

a:
- “la red usa comunicación de forma causal y medible”.

## Roadmap concreto

### 1. Romper la simetría del escenario

Objetivo:
- evitar que ambos aprendan una trayectoria fija y lleguen casi al mismo tiempo.

Acciones:
- randomizar posiciones iniciales,
- randomizar orientaciones iniciales,
- mantener restricciones mínimas para que no nazcan dentro de la U o encima de obstáculos.

Resultado esperado:
- uno de los agentes llegará antes en más episodios,
- habrá oportunidad real para que el otro use la señal.

---

### 2. Diseñar escenarios donde la comunicación sí sea útil

Objetivo:
- que el segundo agente no pueda resolver tan fácil solo con navegación reactiva o memoria de trayectorias.

Acciones:
- crear configuraciones donde el emisor vea/alcance la meta antes que el receptor,
- hacer que el receptor quede más lejos, peor orientado o con peor acceso,
- usar variaciones de la U o del obstáculo simple.

Resultado esperado:
- la señal social podría aportar información útil y no redundante.

---

### 3. Hacer ablaciones causales

Objetivo:
- comprobar si la comunicación realmente cambia el comportamiento.

Mediciones clave:
- con comunicación normal,
- sin comunicación,
- con señal falsa o permutada,
- con el emisor silenciado.

Resultado esperado:
- si la política usa comunicación, el rendimiento debe caer cuando se elimina o corrompe la señal.

---

### 4. Cambiar el criterio de éxito experimental

Objetivo:
- no evaluar solo “llegó o no llegó”.

Agregar métricas como:
- tiempo entre llegada del primer agente y llegada del segundo,
- cantidad y duración de vocalizaciones,
- cambio de trayectoria del receptor tras la vocalización,
- diferencia de rendimiento con vs sin señal.

Resultado esperado:
- poder argumentar utilidad de la vocalización, no solo éxito final.

---

### 5. Ajustar el fitness para favorecer comunicación útil

Objetivo:
- que evolución no solo premie navegación individual fuerte.

Idea general:
- seguir premiando éxito,
- pero agregar presión selectiva para que la señal produzca ventaja real en el compañero,
- por ejemplo, premiar mejora del segundo agente cuando el primero ya encontró la meta.

Resultado esperado:
- la evolución tendrá incentivo para desarrollar señales funcionales.

---

### 6. Separar preguntas experimentales

Plantear dos preguntas distintas:

1. **¿Aprenden a navegar?**
   - ya casi respondida: sí.

2. **¿Aprenden a comunicarse?**
   - todavía no respondida.

Resultado esperado:
- evitar confundir navegación robusta con comunicación emergente.

## Diagnóstico honesto

### Lo que ya se logró
- una política robusta de navegación,
- resolución estable de la U,
- resolución estable del obstáculo simple,
- entrenamiento funcional con GRU residual.

### Lo que todavía falta demostrar
- que la vocalización sea necesaria o útil,
- que el receptor use la señal de forma causal,
- que el sistema no dependa de una trayectoria memorizada en spawns fijos.

## Prioridad inmediata

Si solo se hicieran tres cosas después de esto, deberían ser:

1. randomizar spawns,
2. correr ablaciones causales de la señal,
3. redefinir fitness/métricas para detectar utilidad de comunicación.

## Veredicto

Sí es posible desarrollar la parte de comunicación para el paper.

Pero el siguiente avance ya no pasa por mejorar más la navegación heurística, sino por:
- romper la simetría,
- medir causalidad,
- y seleccionar explícitamente por utilidad comunicativa.

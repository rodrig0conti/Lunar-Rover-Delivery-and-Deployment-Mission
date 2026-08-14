# Lunar Delivery System — Simulador de descenso

Simulador modular para las fases 1–4 de la misión (órbita inicial, preparación
del descenso, descenso propulsado, aterrizaje suave). Herramienta de
dimensionamiento y verificación por análisis, no diseño detallado de GNC.

**Estado actual: Paso 1** — cuerpo central con rotación y gravedad, propagación
orbital, visualización y verificación. Sin encendido de motor todavía.

---

## Arquitectura

| Módulo | Responsabilidad | Depende de |
|---|---|---|
| `config.py` | Constantes físicas y configuración de misión. No calcula nada. | — |
| `moon.py` | Cuerpo central: gravedad (punto + J2), rotación, marcos MCI/MCMF, geometría de superficie. | `config` |
| `spacecraft.py` | Vehículo: estado, masas, modelo de propulsión, Tsiolkovsky, TWR. | `config` |
| `mathutils.py` | Numérica pura: RK4, RKF45, conversiones estado ↔ elementos. No sabe nada de la Luna. | — |
| `dynamics.py` | Ensambla dy/dt. Único sitio donde se suman aceleraciones. | `moon`, `spacecraft` |
| `simulation.py` | Bucle temporal + `History` con las series derivadas. | `mathutils` |
| `visualization.py` | Figuras estáticas y animación 3D. | `moon`, `config` |
| `main.py` | Punto de entrada: monta, propaga, dibuja. | todo |
| `validate.py` | Suite de verificación del propagador. | todo |

La regla que mantiene esto limpio: **`mathutils` no importa `moon`, y `moon` no
importa `spacecraft`.** Las dependencias van en una sola dirección, así que
cualquier módulo se puede testear aislado.

## Convenciones

- **Unidades:** km, s, kg, rad. Grados solo en la interfaz de usuario.
- **Estado:** `y = [rx, ry, rz, vx, vy, vz, m]` (7 elementos, marco MCI).
  La masa va **dentro** del estado integrado porque durante un encendido la
  aceleración y el gasto másico están acoplados.
- **Marcos:** MCI (inercial, donde se integra) y MCMF (fijo al cuerpo, donde
  viven el sitio de aterrizaje y la traza). Coinciden en t = 0.

## Uso

```bash
python main.py                          # 3 órbitas, figuras en pantalla
python main.py --orbits 10              # más largo
python main.py --animate                # animación
python main.py --animate --spin-exaggeration 200 --save orbit.gif
python main.py --no-j2                  # caso de validación dos cuerpos
python main.py --integrator rk45        # paso adaptativo
python validate.py                      # informe de verificación
```

> `--spin-exaggeration` multiplica solo la rotación **mostrada**. La dinámica y
> la traza usan siempre la velocidad real. Déjalo en 1.0 para el informe.

## Verificación (`validate.py`)

20 comprobaciones, todas superadas:

- Ida y vuelta elementos ↔ estado.
- Conservación de energía y momento angular en dos cuerpos: error relativo
  ~1e-15, es decir precisión de máquina. Confirma que el error de integración
  es despreciable frente al error de modelado.
- Periodo y velocidad circular contra la solución analítica.
- **Regresión nodal por J2** contra la tasa secular
  `dΩ/dt = -1.5 J2 n (R/p)² cos i` — coincide al 0.08 %.
- Transformaciones de marco: inversas exactas, norma preservada, y un punto de
  superficie permanece fijo en MCMF.

Esta tabla vale directamente como anexo del informe: convierte el simulador en
un método de verificación admisible en vez de una figura bonita.

## Notas de modelado

- **J2 = 2.033e-4.** Es lo que hace precesar el plano orbital y la razón de
  usar i ≈ 86° en vez de 90° exactos. Con `--no-j2` se recupera el problema de
  dos cuerpos puro, que es el caso con solución cerrada.
- **El diagnóstico de energía** usa solo el potencial de punto másico, así que
  con J2 activo oscila (~5e-4). Eso es física real, no error numérico: para
  medir calidad de integración usa `--no-j2`.
- **Rotación lunar:** periodo sidéreo 27.32 días. En 3 órbitas (~6 h) la Luna
  gira 0.8°. Real pero invisible; de ahí la exageración de visualización.

## Siguiente paso

Fase 2: capa de guiado y secuenciador de fases. La estructura ya está
preparada — `Spacecraft.set_command()` existe y `Dynamics` ya suma el empuje;
falta el módulo `guidance.py` que decida `throttle` y dirección, y un
`phases.py` con las condiciones de transición (separación → deorbit burn →
coast → PDI).

"""
=============================================================================
PRÁCTICA DE MÉTODOS NUMÉRICOS II
Interpolación del Consumo de Combustible y Autonomía de Vehículos
=============================================================================
Autores: Antonio Calvache Vidal, Carlos Frontera Sureda, Neus Ramón Calafell
Curso  : Métodos Numéricos II
=============================================================================
Las funciones de interpolación (Polinomio_Lagrange, Polinomio_Interpolacion,
Interpolacion_Newton, SplineNatural) y las de Transformada Rápida de Fourier
(TransformadaFourierRapida, SubirNivel, TransformacionIntervalo, ValoresFFT)
son los códigos oficiales de la asignatura.
=============================================================================
"""

import numpy as np
import math
import cmath
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sp.init_printing()

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN GLOBAL DE MATPLOTLIB
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "lines.linewidth": 1.8,
})

# ---------------------------------------------------------------------------
# 1. DATOS EXPERIMENTALES
# ---------------------------------------------------------------------------
v_datos = [30.0, 50.0, 70.0, 90.0, 110.0, 130.0]      # km/h
C_datos = [8.5, 6.2, 5.5, 5.8, 6.5, 7.6]              # L / 100 km
DEPOSITO = 50.0                                        # L
A_datos = [(DEPOSITO / c) * 100.0 for c in C_datos]   # km

# Wrapper: las funciones oficiales reciben f como callable, no listas.
_lookup = dict(zip(v_datos, C_datos))
def consumo(v):
    """Función f(v_i) = C_i evaluada en los nodos experimentales."""
    return _lookup[float(v)]

print("=" * 60)
print("DATOS EXPERIMENTALES")
print("=" * 60)
print(f"{'v (km/h)':>10} {'C (L/100km)':>14} {'Autonomía (km)':>16}")
print("-" * 42)
for v, c, a in zip(v_datos, C_datos, A_datos):
    print(f"{v:>10.1f} {c:>14.2f} {a:>16.1f}")
print()

# ===========================================================================
# 2. CÓDIGOS OFICIALES DE CLASE — INTERPOLACIÓN DE LAGRANGE
# ===========================================================================
def Polinomio_Lagrange(k, nodos_originales, expanded=True):
    """
    Dados n+1 nodos x_i, i=0,1,...,n y un indice k,
    devuelve el polinomio de Lagrange que corresponde al índice k.

    Parámetros:
    * nodos_originales: vector de nodos.
    * k: índice del punto del cual queremos calcular su polinomio de Lagrange.

    Valor de retorno:
    * Lpoli: Polinomio de Lagrange de los n+1 puntos en el índice k.
    """
    a = nodos_originales.copy()
    den = 1
    for j in a:
        if j != a[k]:
            den = den * (a[k] - j)
    a.remove(a[k])
    x = sp.symbols('x')
    Lpoli = 1
    for j in a:
        Lpoli = Lpoli * (x - j)
    Lpoli = Lpoli / den
    if expanded:
        return sp.expand(Lpoli)
    else:
        return Lpoli


def Polinomio_Interpolacion(nodos, f=consumo):
    """
    Dados n+1 nodos x_i, i=0,1,...,n y una función f,
    devuelve el polinomio interpolador de Lagrange de f
    que pasa por los puntos (x_i, f(x_i)).
    """
    y = []
    for i in nodos:
        y.append(f(i))
    x = sp.symbols('x')
    Lpoli = 0
    for i in range(len(nodos)):
        Lpoli += y[i] * Polinomio_Lagrange(i, nodos)
    return sp.expand(Lpoli)


# ===========================================================================
# 3. CÓDIGOS OFICIALES DE CLASE — INTERPOLACIÓN DE NEWTON
# ===========================================================================
def Interpolacion_Newton(nodos, f=consumo):
    """
    Dados n+1 nodos x_i, i=0,1,...,n y una función f, devuelve el polinomio
    interpolador de f que pasa por los puntos (x_i, f(x_i)) usando el método
    de las diferencias divididas de Newton.
    """
    x = sp.symbols('x')
    Q = []
    for i in range(len(nodos)):
        Q.append([])
    for i in range(len(nodos)):
        Q[i].append(f(nodos[i]))
    for i in range(len(nodos)):
        for j in range(1, i + 1):
            Q[i].append((Q[i][j-1] - Q[i-1][j-1]) / (nodos[i] - nodos[i-j]))
    pol = Q[0][0]
    for i in range(1, len(nodos)):
        producto = Q[i][i]
        for j in range(i):
            producto = producto * (x - nodos[j])
        pol = pol + producto
    return pol, Q


# ===========================================================================
# 4. CÓDIGOS OFICIALES DE CLASE — SPLINE CÚBICO NATURAL
# ===========================================================================
def SplineNatural(nodos, f=consumo, verbose=True):
    """
    Dados n+1 puntos en el eje de las x indexados y una función f,
    devuelve el spline cúbico natural que interpola f.

    Cada tramo es:
        S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
    """
    n = len(nodos) - 1
    a = []
    b = []
    c = [0]   # condición natural en x_0
    d = []
    h = []
    for i in range(n):
        h.append(nodos[i+1] - nodos[i])
    for i in range(n+1):
        a.append(f(nodos[i]))

    # Sistema tridiagonal para los c_i interiores
    y = []
    for i in range(1, n):
        y.append((3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1]))

    A = np.zeros((n-1, n-1))
    A[0][0] = 2*(h[0]+h[1])
    A[0][1] = h[1]
    for i in range(1, n-2):
        A[i][i-1] = h[i]
        A[i][i] = 2*(h[i] + h[i+1])
        A[i][i+1] = h[i+1]
    A[n-2][n-3] = h[n-2]
    A[n-2][n-2] = 2*(h[n-2] + h[n-1])

    sol_Ay = list(np.linalg.solve(A, y))
    for i in sol_Ay:
        c.append(i)
    c.append(0)   # condición natural en x_n

    for i in range(n):
        b.append((1/h[i])*(a[i+1]-a[i]) - (h[i]/3)*(c[i+1] + 2*c[i]))
        d.append((1/(3*h[i]))*(c[i+1] - c[i]))

    x = sp.symbols('x')
    polsplines = []
    for i in range(n):
        polsplines.append(sp.expand(
            a[i] + b[i]*(x - nodos[i]) + c[i]*(x - nodos[i])**2 + d[i]*(x - nodos[i])**3
        ))

    if verbose:
        for k, spline in enumerate(polsplines):
            print("Si x en [{},{}] => {}".format(nodos[k], nodos[k+1], spline))

    return polsplines, a, b, c, d


# ===========================================================================
# 5. CÓDIGOS OFICIALES DE CLASE — TRANSFORMADA RÁPIDA DE FOURIER
# ===========================================================================
def SubirNivel(y, j, c):
    """
    Subrutina del algoritmo FFT: combina coeficientes de nivel inferior.
    """
    L = c.shape[0]
    cn = np.zeros(L, dtype=np.csingle)
    z = np.zeros(L)
    for k in range(math.floor(L / 2)):
        cn[k] = c[k] + cmath.exp(1j * k * cmath.pi / math.pow(2, j)) * c[k + math.floor(L / 2)]
        cn[k + math.floor(L / 2)] = c[k] - cmath.exp(1j * k * cmath.pi / math.pow(2, j)) * c[k + math.floor(L / 2)]
    for i in range(math.floor(L/2)):
        z[2*i] = y[i]
        z[2*i+1] = y[math.floor(L / 2)+i]
    return (z, cn)


def TransformadaFourierRapida(m, y, verbose=False):
    """
    Calcula los coeficientes de la aproximación FFT.

    Args:
        m: (int) Potencia de 2.
        y: (ndarray) Muestra inicial de tamaño 2m.

    Returns:
        (resultado, a, b): coeficientes c_k, a_k y b_k de la serie discreta.
    """
    p = math.floor(math.log(m, 2))
    z = y.copy()

    for i in range(1, p + 1):
        for j in range(1, int(math.pow(2, p - 1))+1):
            start = (j - 1) * int(math.pow(2, p + 2 - i))
            end = j * int(math.pow(2, p + 2 - i))
            aux = z[start : end].copy()
            aux = np.concatenate((np.array([aux[i] for i in range(0, aux.shape[0], 2)]),
                                  np.array([aux[i] for i in range(1, aux.shape[0], 2)])))
            z[start : (end - 1)] = aux[0 : int(math.pow(2, p + 2 - i)) - 1].copy()

    y = z.copy()
    c = np.zeros(2 * m, dtype=np.csingle)

    for i in range(m):
        c[2 * i] = y[2 * i] + y[2 * i + 1]
        c[2 * i + 1] = y[2 * i] - y[2 * i + 1]

    resultado = [0] * int(math.pow(2, p - 1))
    for j in range(1, int(math.pow(2, p - 1)) + 1):
        resultado[j - 1] = SubirNivel(
            y[((j - 1) * int(math.pow(2, 2))) : (j * int(math.pow(2, 2)))], 1,
            c[((j-1) * int(math.pow(2, 2))):(j * int(math.pow(2, 2)))]
        )

    for i in range(2, p + 1):
        resultadoN = [0] * int(math.pow(2, p - i))
        for j in range(int(math.pow(2, p - i))):
            yj = np.concatenate((resultado[2 * j][0], resultado[2 * j + 1][0]))
            cj = np.concatenate((resultado[2 * j][1], resultado[2 * j + 1][1]))
            resultadoN[j] = SubirNivel(yj, i, np.array(cj))
        resultado = resultadoN

    resultado = resultado[0][1]

    a_coef = np.array([math.pow(-1, k) / m * resultado[k].real for k in range(0, m + 1)])
    b_coef = np.array([0 if np.isclose(0, math.pow(-1, k) / m * resultado[k].imag)
                       else math.pow(-1, k) / m * resultado[k].imag
                       for k in range(1, m)])
    return (resultado, a_coef, b_coef)


def TransformacionIntervalo(t, start, end):
    """
    Cambio de variable [start, end] -> [-pi, pi].
    """
    x = (2 * math.pi) / (end - start) * (t - (start + end) / 2)
    return x


def ValoresFFT(a, b, x, n):
    """
    Evalúa la serie de Fourier discreta S_n(x) en un array de puntos.
    """
    if a.shape[0] != n + 1 and b.shape[0] != n - 1:
        print("La longitud de los arrays de coeficientes no es correcta")
        return
    m = x.shape[0]
    aprox = np.zeros(m)
    for i in range(m):
        aprox[i] = (a[0] + a[n] * math.cos(n * x[i])) / 2
        for j in range(1, n):
            aprox[i] += a[j] * math.cos(j * x[i]) + b[j-1] * math.sin(j * x[i])
    return aprox


# ===========================================================================
# 6. UTILIDADES NUMÉRICAS
# ===========================================================================
def lambdify_pol(pol):
    """Convierte una expresión simbólica en una función NumPy-callable."""
    x = sp.symbols('x')
    return sp.lambdify(x, pol, 'numpy')


def evaluar_spline(polsplines, nodos, x_eval):
    """Evalúa el spline cúbico (lista de tramos) en una malla."""
    funs = [lambdify_pol(p) for p in polsplines]
    res = np.zeros_like(x_eval, dtype=float)
    for k, xv in enumerate(x_eval):
        i = int(np.searchsorted(nodos, xv, side='right') - 1)
        i = max(0, min(i, len(funs) - 1))
        res[k] = funs[i](xv)
    return res


def nodos_chebyshev(a, b, n):
    """n+1 nodos de Chebyshev trasladados al intervalo [a, b]."""
    k = np.arange(0, n + 1)
    t_k = np.cos((2*k + 1) / (2*n + 2) * np.pi)
    return np.sort((b - a) / 2.0 * t_k + (a + b) / 2.0)


# ===========================================================================
# 7. APLICACIÓN A LOS DATOS DE CONSUMO
# ===========================================================================
v_fino = np.linspace(30.0, 130.0, 500)

# ---- Lagrange ----
pol_lagrange = Polinomio_Interpolacion(v_datos, f=consumo)
fun_lagrange = lambdify_pol(pol_lagrange)
C_lagrange = fun_lagrange(v_fino)

idx_min_lag = int(np.argmin(C_lagrange))
v_opt_lag = v_fino[idx_min_lag]
C_opt_lag = C_lagrange[idx_min_lag]

print("=" * 60)
print("INTERPOLACIÓN DE LAGRANGE")
print("=" * 60)
print(f"Polinomio: {sp.nsimplify(pol_lagrange, rational=False, tolerance=1e-12)}")
print(f"Velocidad de consumo mínimo : {v_opt_lag:.2f} km/h")
print(f"Consumo mínimo interpolado  : {C_opt_lag:.4f} L/100 km")
print(f"Autonomía máxima estimada   : {(DEPOSITO/C_opt_lag)*100:.1f} km")
print()

# ---- Newton ----
pol_newton, Q = Interpolacion_Newton(v_datos, f=consumo)
fun_newton = lambdify_pol(pol_newton)
C_newton = fun_newton(v_fino)

print("=" * 60)
print("INTERPOLACIÓN DE NEWTON (diferencias divididas)")
print("=" * 60)
print("Tabla de diferencias divididas Q[i][j]:")
print(pd.DataFrame(Q).replace(np.nan, ''))
print("\nCoeficientes de Newton (diagonal Q[i][i] = f[v_0,...,v_i]):")
for i in range(len(v_datos)):
    print(f"  f[v_0,...,v_{i}] = {float(Q[i][i]):+.6e}")

idx_min_newt = int(np.argmin(C_newton))
v_opt_newt = v_fino[idx_min_newt]
C_opt_newt = C_newton[idx_min_newt]
print(f"\nVelocidad de consumo mínimo : {v_opt_newt:.2f} km/h")
print(f"Consumo mínimo interpolado  : {C_opt_newt:.4f} L/100 km")
print(f"Autonomía máxima estimada   : {(DEPOSITO/C_opt_newt)*100:.1f} km")
print(f"Diferencia máx. vs Lagrange : {np.max(np.abs(C_lagrange - C_newton)):.2e}")
print()

# ---- Spline natural ----
print("=" * 60)
print("SPLINE CÚBICO NATURAL")
print("=" * 60)
polsplines, a_sp, b_sp, c_sp, d_sp = SplineNatural(v_datos, f=consumo, verbose=True)
print("\nCoeficientes (a_i, b_i, c_i, d_i):")
print(f"{'i':>2} {'v_i':>6} {'a_i':>10} {'b_i':>14} {'c_i':>14} {'d_i':>14}")
for i in range(len(v_datos)-1):
    print(f"{i:>2} {v_datos[i]:>6.0f} {a_sp[i]:>10.4f} "
          f"{b_sp[i]:>+14.6e} {c_sp[i]:>+14.6e} {d_sp[i]:>+14.6e}")

C_spline = evaluar_spline(polsplines, v_datos, v_fino)
idx_min_sp = int(np.argmin(C_spline))
v_opt_sp = v_fino[idx_min_sp]
C_opt_sp = C_spline[idx_min_sp]
print(f"\nVelocidad de consumo mínimo : {v_opt_sp:.2f} km/h")
print(f"Consumo mínimo interpolado  : {C_opt_sp:.4f} L/100 km")
print(f"Autonomía máxima estimada   : {(DEPOSITO/C_opt_sp)*100:.1f} km")
print()

# ---- Nodos de Chebyshev (comparativa Runge) ----
v_cheby = nodos_chebyshev(30.0, 130.0, 5)
C_cheby = evaluar_spline(polsplines, v_datos, v_cheby)   # spline como referencia
pol_newton_cheby, _ = Interpolacion_Newton(list(v_cheby),
                                           f=lambda v: float(evaluar_spline(polsplines,
                                                                            v_datos,
                                                                            np.array([v]))[0]))
fun_newton_cheby = lambdify_pol(pol_newton_cheby)
C_newton_cheby = fun_newton_cheby(v_fino)

print("=" * 60)
print("NODOS DE CHEBYSHEV (grado 5) — comparativa Runge")
print("=" * 60)
for k, vk in enumerate(v_cheby):
    print(f"  x_{k} = {vk:.4f} km/h")
print()

# ---- Análisis del error ----
error_eq    = np.abs(C_spline - C_newton)
error_cheby = np.abs(C_spline - C_newton_cheby)
print("=" * 60)
print("ANÁLISIS DE ERROR (vs. spline cúbico como referencia)")
print("=" * 60)
print(f"Error máx. Newton equiespaciados : {np.max(error_eq):.4e}")
print(f"Error máx. Newton Chebyshev      : {np.max(error_cheby):.4e}")
print(f"Reducción de error               : x{np.max(error_eq)/max(np.max(error_cheby),1e-16):.1f}")
print()

# ---- Autonomía interpolada ----
A_spline = (DEPOSITO / C_spline) * 100.0
A_newton = (DEPOSITO / C_newton) * 100.0
print("=" * 60)
print("AUTONOMÍA INTERPOLADA (spline cúbico)")
print("=" * 60)
print(f"{'v (km/h)':>10} {'C interp. (L/100km)':>22} {'Autonomía (km)':>16}")
print("-" * 50)
for vi in [60, 70, 80, 100, 120]:
    ci = float(evaluar_spline(polsplines, v_datos, np.array([vi]))[0])
    ai = (DEPOSITO / ci) * 100.0
    print(f"{vi:>10} {ci:>22.4f} {ai:>16.1f}")
print()

# ---- FFT con la función oficial ----
m_fft = 32
N_fft = 2 * m_fft   # 64 muestras
v_fft = np.linspace(30.0, 130.0, N_fft, endpoint=False)
C_fft = evaluar_spline(polsplines, v_datos, v_fft)
x_fft = TransformacionIntervalo(v_fft, 30.0, 130.0)   # paso a [-pi, pi]

resultado_fft, a_fft, b_fft = TransformadaFourierRapida(m_fft, C_fft, verbose=False)

# Amplitud por armónico (raíz de a_k^2 + b_{k-1}^2)
amp = np.zeros(m_fft + 1)
amp[0] = abs(a_fft[0]) / 2.0
for k in range(1, m_fft):
    amp[k] = math.sqrt(a_fft[k]**2 + b_fft[k-1]**2)
amp[m_fft] = abs(a_fft[m_fft]) / 2.0

print("=" * 60)
print("ANÁLISIS ESPECTRAL (FFT oficial, m={}, 2m={} muestras)".format(m_fft, N_fft))
print("=" * 60)
print("Primeros 8 armónicos:")
print(f"{'k':>4} {'Amplitud':>12}")
print("-" * 20)
for k in range(8):
    print(f"{k:>4} {amp[k]:>12.6f}")

energia_total = float(np.sum(amp**2))
energia_3 = float(np.sum(amp[:3]**2))
print(f"\nEnergía total del espectro   : {energia_total:.4f}")
print(f"Energía primeros 3 armónicos : {energia_3:.4f} ({100*energia_3/energia_total:.1f}%)")
print()

# Reconstrucción para visualizar
C_fft_reconstruido = ValoresFFT(a_fft, b_fft, x_fft, m_fft)

# ===========================================================================
# 8. FIGURAS
# ===========================================================================
v_arr = np.array(v_datos)
C_arr = np.array(C_datos)
A_arr = np.array(A_datos)

# ---- FIGURA 1: Comparativa de métodos de interpolación ----
fig1, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(v_fino, C_newton, color='crimson', linestyle='--',
        label='Newton / Lagrange (grado 5)', linewidth=1.8)
ax.plot(v_fino, C_spline, color='royalblue', linestyle='-',
        label='Spline cúbico natural', linewidth=2.0)
ax.scatter(v_arr, C_arr, color='black', zorder=5, s=60, label='Datos experimentales')
ax.axvline(v_opt_sp, color='royalblue', linestyle=':', alpha=0.6)
ax.axhline(C_opt_sp, color='royalblue', linestyle=':', alpha=0.6)
ax.annotate(f'$v^* \\approx {v_opt_sp:.0f}$ km/h\n$C^* \\approx {C_opt_sp:.2f}$ L/100km',
            xy=(v_opt_sp, C_opt_sp), xytext=(v_opt_sp + 12, C_opt_sp + 0.4),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=9)
ax.set_xlabel('Velocidad (km/h)')
ax.set_ylabel('Consumo (L / 100 km)')
ax.set_title('Interpolación del consumo')
ax.legend()
ax.grid(True, alpha=0.35)
ax.set_xlim(25, 135)

ax2 = axes[1]
ax2.plot(v_fino, C_newton_cheby, color='seagreen', linestyle='-.',
         label='Newton (nodos Chebyshev)', linewidth=1.8)
ax2.plot(v_fino, C_newton, color='crimson', linestyle='--',
         label='Newton (nodos equiespaciados)', linewidth=1.8)
ax2.plot(v_fino, C_spline, color='royalblue', linestyle='-',
         label='Spline cúbico natural', linewidth=2.0, alpha=0.7)
ax2.scatter(v_arr, C_arr, color='black', zorder=5, s=60)
ax2.set_xlabel('Velocidad (km/h)')
ax2.set_ylabel('Consumo (L / 100 km)')
ax2.set_title('Nodos equiespaciados vs. Chebyshev')
ax2.legend()
ax2.grid(True, alpha=0.35)
ax2.set_xlim(25, 135)

fig1.suptitle('Figura 1 — Comparativa de Métodos de Interpolación', fontsize=13, y=1.01)
fig1.tight_layout()
fig1.savefig('figura1_comparativa.png', bbox_inches='tight', dpi=130)
print("Figura 1 guardada: figura1_comparativa.png")

# ---- FIGURA 2: Autonomía interpolada ----
fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(v_fino, A_spline, color='darkorange', linewidth=2.2, label='Autonomía (spline cúbico)')
ax.plot(v_fino, A_newton, color='steelblue', linestyle='--', linewidth=1.6,
        label='Autonomía (Newton grado 5)')
ax.scatter(v_arr, A_arr, color='black', zorder=5, s=60, label='Datos experimentales')
idx_max_a = int(np.argmax(A_spline))
ax.axvline(v_fino[idx_max_a], color='darkorange', linestyle=':', alpha=0.7)
ax.annotate(f'Máx: {A_spline[idx_max_a]:.0f} km\n@ {v_fino[idx_max_a]:.0f} km/h',
            xy=(v_fino[idx_max_a], A_spline[idx_max_a]),
            xytext=(v_fino[idx_max_a] + 10, A_spline[idx_max_a] - 50),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=9)
ax.set_xlabel('Velocidad (km/h)')
ax.set_ylabel('Autonomía (km)')
ax.set_title('Figura 2 — Autonomía estimada en función de la velocidad\n(depósito = 50 L)')
ax.legend()
ax.grid(True, alpha=0.35)
ax.set_xlim(25, 135)
fig2.tight_layout()
fig2.savefig('figura2_autonomia.png', bbox_inches='tight', dpi=130)
print("Figura 2 guardada: figura2_autonomia.png")

# ---- FIGURA 3: Espectro FFT (oficial) ----
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))

ax3a = axes3[0]
ax3a.plot(v_fft, C_fft, color='royalblue', linewidth=1.8, label='Señal remuestreada')
ax3a.plot(v_fft, C_fft_reconstruido, color='darkorange', linestyle='--',
          linewidth=1.5, label='Reconstrucción FFT $S_m(x)$')
ax3a.scatter(v_arr, C_arr, color='crimson', zorder=5, s=50, label='Datos originales')
ax3a.set_xlabel('Velocidad (km/h)')
ax3a.set_ylabel('Consumo (L / 100 km)')
ax3a.set_title(f'Señal remuestreada ({N_fft} puntos) y reconstrucción')
ax3a.legend(fontsize=9)
ax3a.grid(True, alpha=0.35)

ax3b = axes3[1]
n_show = 16
ax3b.bar(range(n_show), amp[:n_show], color='steelblue', alpha=0.8, edgecolor='navy')
ax3b.set_xlabel('Número de armónico $k$')
ax3b.set_ylabel('Amplitud')
ax3b.set_title('Espectro de amplitudes (FFT oficial)')
ax3b.set_xlim(-0.5, n_show - 0.5)
for k in range(4):
    ax3b.text(k, amp[k] + 0.05, f'{amp[k]:.3f}', ha='center', fontsize=8)
ax3b.grid(True, alpha=0.35, axis='y')

fig3.suptitle('Figura 3 — Análisis Espectral (TransformadaFourierRapida)', fontsize=13)
fig3.tight_layout()
fig3.savefig('figura3_fft.png', bbox_inches='tight', dpi=130)
print("Figura 3 guardada: figura3_fft.png")

# ---- FIGURA 4: Error de interpolación ----
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))

ax4a = axes4[0]
ax4a.semilogy(v_fino, error_eq + 1e-16, color='crimson', linestyle='--',
              label='Newton equiespaciados')
ax4a.semilogy(v_fino, error_cheby + 1e-16, color='seagreen',
              label='Newton Chebyshev')
ax4a.set_xlabel('Velocidad (km/h)')
ax4a.set_ylabel('Error absoluto (escala logarítmica)')
ax4a.set_title('Error vs. spline cúbico (referencia)')
ax4a.legend()
ax4a.grid(True, alpha=0.35, which='both')

ax4b = axes4[1]
ax4b.plot(v_fino, error_eq, color='crimson', linestyle='--',
          label=f'Equiespaciados (máx: {np.max(error_eq):.4f})')
ax4b.plot(v_fino, error_cheby, color='seagreen',
          label=f'Chebyshev (máx: {np.max(error_cheby):.4f})')
ax4b.scatter(v_arr, np.zeros_like(v_arr), marker='|', color='black',
             s=200, zorder=5, label='Nodos equiespaciados')
ax4b.scatter(v_cheby, np.zeros_like(v_cheby), marker='x', color='seagreen',
             s=100, zorder=5, label='Nodos Chebyshev')
ax4b.set_xlabel('Velocidad (km/h)')
ax4b.set_ylabel('Error absoluto')
ax4b.set_title('Distribución del error en escala lineal')
ax4b.legend(fontsize=8)
ax4b.grid(True, alpha=0.35)

fig4.suptitle('Figura 4 — Análisis del Error de Interpolación', fontsize=13)
fig4.tight_layout()
fig4.savefig('figura4_error.png', bbox_inches='tight', dpi=130)
print("Figura 4 guardada: figura4_error.png")

print("\n" + "=" * 60)
print("EJECUCIÓN COMPLETADA CORRECTAMENTE")
print("=" * 60)
"""
Este archivo contiene el código para la comprensión y aproximación
de la solución del sistema Lotka-Volterra utilizando el método
de Runge-Kutta de 4to Orden.
"""

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from autoclass import autoargs
from scipy import integrate
sns.set_style("darkgrid")
np.random.seed(23)


class LotkaVolterraModel:
    """
    El sistema de ecuaciones diferenciales ordinarias,
    más conocido como el Modelo de Lotka-Volterra.
    Notación: `dF_dx -> Derivada de F sobre x`

    Parámetros:
        float alpha: Tasa de natalidad (presas)
        float delta: Tasa de mortalidad (presas)
        float gamma: Tasa de natalidad (depredadores)
        float delta: Tasa de mortalidad (depredadores)
    """

    @autoargs
    def __init__(self, alpha, beta, gamma, delta):
        self.dP_dt = lambda p, d: (alpha * p) - (beta * p * d)
        self.dD_dt = lambda p, d: (delta * p * d) - (gamma * d)

    """ Evaluación """
    def __call__(self, count):
        return np.array((
            self.dP_dt(count[0], count[1]),
            self.dD_dt(count[0], count[1])
        ), np.float32)


def runge_kutta_4(model, p0, trange=(0, 200), h=0.05):
    """
    Algoritmo Runge-Kutta de 4to Orden.
    Parametros:
        callable model: Modelo a evaluar
        (float, float) p0: Punto (x, y) inicial
        (float, float) trange: Rango temporal a utilizar
        float h: Paso de integración
    Devuelve:
        np.array(np.float32) X: Aproximaciones de X
        np.array(np.float32) Y: Aproximaciones de Y
        np.array(np.float32) T: Puntos temporales utilizados
    """
    # inicializamos
    T = np.arange(*trange, h, np.float32)
    x, y = p0
    r = np.array([x, y], np.float32)
    X, Y = np.array([], np.float32), np.array([], np.float32)

    for t in T:
        # actualizamos los puntos
        X = np.append(X, r[0])
        Y = np.append(Y, r[1])

        # paso runge-kutta
        k1 = h * model(r)
        k2 = h * model(r + 0.5 * k1)
        k3 = h * model(r + 0.5 * k2)
        k4 = h * model(r + k3)
        r += (k1 + 2 * k2 + 2 * k3 + k4)/6

    return X, Y, T


def plot_evolution(model, p0, trange, h):
    """
    Gráfico de la aproximación de la evolución de un modelo
    a través del tiempo, utilizando Runge Kutta de 4to orden.

    Parámetros:
        callable model: Modelo a evaluar
        (float, float) p0: Punto (x, y) inicial
        (float, float) trange: Rango temporal a utilizar
        float h: Paso de integración
    """

    # aproximamos la evolución con rk4
    P, D, T = runge_kutta_4(
        model,
        p0=p0,
        trange=trange,
        h=h
    )

    # inicializamos el gráfico
    f, ax = plt.subplots(1, figsize=(10, 8))
    ax.set_title('''
Evolución a través del tiempo de Lotka-Volterra
   con p0=(40, 9), t en (0, 200) y h = 0.05
    ''')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Población')

    # graficamos los resultados a través del tiempo
    ax.plot(T, P, label='Presas')
    ax.plot(T, D, label='Depredadores')

    ax.legend()
    plt.show()


def plot_df(model, fases, n_vect):
    """
    Gráfico del diagrama de fases en conjunto con el diagrama de dirección.
    Parámetros:
        callable model: Modelo a evaluar
        np.array(np.float32) fases: puntos iniciales para cada fase
        int n_vect: vectores de dirección a mostrar
    """

    # inicializamos el gráfico
    f, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title(f'Diagrama de fases')
    ax.set_xlabel('Población de presas')
    ax.set_ylabel('Población de depredadores')

    # aproximamos el comportamiento dados estos puntos
    # utilizando Runge Kutta y los graficamos
    for p0 in fases:
        X, Y, _ = runge_kutta_4(model=model, p0=p0)
        ax.plot(X, Y, label=f'p0 = ({p0[0]:.2f}, {p0[1]:.2f})')

    # construimos una grilla sobre los límites del gráfico
    # con la cantidad de vectores definida
    X, Y = np.meshgrid(
        np.linspace(0, plt.xlim(xmin=0)[1], n_vect),
        np.linspace(0, plt.ylim(ymin=0)[1], n_vect)
    )

    # computamos la tasa de crecimiento sobre la grilla
    DX, DY = model((X, Y))

    # normalizamos la tasa y evitamos división por cero
    M = (np.hypot(DX, DY))
    M[M == 0] = 1.
    DX /= M
    DY /= M

    # graficamos los vectores
    Q = plt.quiver(X, Y, DX, DY, M, pivot='mid')

    # mostramos el gráfico
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == '__main__':

    # inicializamos el modelo
    # con los valores requeridos
    model = LotkaVolterraModel(
        alpha=.1,
        beta=.02,
        gamma=.3,
        delta=.01
    )

    # --- EJERCICIO B ---
    fases = np.random.uniform(5, 40, size=(5, 2))

    plot_df(
        model,
        fases=fases,
        n_vect=40
    )

    # --- EJERCICIO C ---
    plot_evolution(
        model,
        p0=(40, 9),
        trange=(0, 200),
        h=0.05
    )

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def specify_initial_and_boundary(mesh_width, mesh_height, initial, initial_derivative, boundary_1, boundary_2):
    """
    Задание граничных и начальных условий
    :param mesh_width: количество ячеек по пространству
    :param mesh_height: количество ячеек по времени
    :param initial: начальное условие (функция от координаты)
    :param initial_derivative: начальное условие на производную по времени (функция от координаты)
    :param boundary_1: граничное условие первого рода на левом конце (функция от времени)
    :param boundary_2: граничное условие первого рода на правом конце (функция от времени)
    :return: массив со значениями температуры (заданы начальные и граничные условия)
    """

    temperature_values = np.empty([mesh_height + 1, mesh_width + 1])
    temperature_derivative_values = np.empty([mesh_height + 1, mesh_width + 1])

    for i in range(mesh_width + 1):
        temperature_values[0, i] = initial(i)
        temperature_values[1, i] = initial_derivative(i) * initial(i)
        temperature_derivative_values[0, i] = initial_derivative(i)

    for k in range(mesh_height + 1):
        temperature_values[k, 0] = boundary_1(k)
        temperature_values[k, mesh_width] = boundary_2(k)

    return temperature_values


def explicit_scheme(temperature_values, time_step, space_step, c_value):
    """
    Реализация явной схемы интегрирования модифицированного уравнения теплопроводности
    :param temperature_values: массив со значениями температуры, в котором заданы начальные и граничные условия
    :param time_step: шаг по времени
    :param space_step: шаг по пространству
    :param c_value: скорость звука в кристалле
    :return: массив со значениями температуры в разных точках кристалла в разные моменты времени
    """
    for n in range(1, temperature_values.shape[0] - 1):
        for m in range(1, temperature_values.shape[1] - 1):
            temperature_values[n + 1][m] = temperature_values[n][m] / (n + 1) - n / (n + 1) * \
                                           (temperature_values[n - 1][m] - 2 * temperature_values[n][m]) + \
                                           c_value ** 2 * n * time_step ** 2 / ((n + 1) * space_step ** 2) * (
                                                   temperature_values[n][m - 1] - 2 * temperature_values[n][m] +
                                                   temperature_values[n][m + 1])

    return temperature_values


def implicit_scheme(temperature_values, time_step, space_step, c_value):
    """
    Реализация неявной схемы интегрирования модифицированного уравнения теплопроводности
    (трёхдиагональная СЛАУ решается методом прогонки)
    :param temperature_values: массив со значениями температуры, в котором заданы начальные и граничные условия
    :param time_step: шаг по времени
    :param space_step: шаг по пространству
    :param c_value: скорость звука в кристалле
    :return: массив со значениями температуры в разных точках кристалла в разные моменты времени
    """
    for n in range(1, temperature_values.shape[0] - 1):
        alpha = np.empty([temperature_values.shape[1]])
        beta = np.empty([temperature_values.shape[1]])

        alpha[0] = 0  # из граничного условия на левом конце
        beta[0] = 0  # из граничного условия на левом конце

        # решаем методом прогонки
        for m in range(1, temperature_values.shape[1]):
            # для модифицированного уравнения теплопроводности
            B = c_value ** 2 / space_step ** 2
            C = (n + 1) / (n * time_step ** 2) + 2 * c_value ** 2 / space_step ** 2
            A = c_value ** 2 / space_step ** 2
            F = ((2 * n + 1) * temperature_values[n][m] - n * temperature_values[n - 1][m]) / (n * time_step ** 2)

            # для волнового уравнения
            # B = 1/space_step**2
            # C = (2*c_value**2*time_step**2+space_step**2)/(space_step**2*c_value**2*time_step**2)
            # A = 1/space_step**2
            # F = (2*T[n][m]-T[n-1][m])/(c_value**2*time_step**2)

            # для уравнения теплопроводности
            # B = time_step/space_step**2
            # C = 1+2*time_step/space_step**2
            # A = time_step/space_step**2
            # F = T[n][m]

            alpha[m] = B / (C - A * alpha[m - 1])
            beta[m] = (A * beta[m - 1] + F) / (C - A * alpha[m - 1])

        for m in range(width, 0, -1):
            temperature_values[n + 1][m - 1] = alpha[m - 1] * temperature_values[n + 1][m] + beta[m - 1]

    return temperature_values


# функции начальных условий и граничных условий первого рода
def gauss(x, alpha=5, median=2.5):
    return np.exp(-alpha * (x * h - median) ** 2)


def one(x):
    return 1


def zero(x):
    return 0


def step(x, l_dist=0.5, a_max=5, full_length=5, mesh_width=1000):
    if (1 / l_dist) / full_length * mesh_width <= x <= (full_length - 1 / l_dist) / full_length * mesh_width:
        return a_max
    else:
        return 0


if __name__ == '__main__':
    c = 1  # скорость звука в кристалле
    total_time = 5  # время протекания процесса
    L = 10  # длина кристалла
    height = 1000  # количество ячеек по времени
    width = 1000  # количество ячеек по пространству

    tau = total_time / height  # шаг по времени
    h = L / width  # шаг по пространству

    print('Шаг по времени: ', tau)
    print('Шаг по пространству', h)

    T_explicit = specify_initial_and_boundary(width, height, step, one, zero, zero)
    T_implicit = specify_initial_and_boundary(width, height, step, one, zero, zero)
    T_explicit = explicit_scheme(T_explicit, tau, h, c)
    T_implicit = implicit_scheme(T_implicit, tau, h, c)

    Length, Time = np.meshgrid(np.linspace(0, L, width + 1), np.linspace(0, total_time, height + 1))

    # 3D графики для явной и неявной схем
    fig1 = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Явная схема', 'Неявная схема')
    )
    fig1.add_trace(
        go.Surface(x=Time, y=Length, z=T_explicit),
        row=1, col=1
    )
    fig1.add_trace(
        go.Surface(x=Time, y=Length, z=T_implicit),
        row=1, col=2
    )
    fig1.show()

    # анимация
    fig2 = go.Figure(
        data=[go.Scatter(
            x=np.array(range(len(T_implicit))) * h,
            y=T_implicit[0],
            mode='lines',
            line={'color': 'blue'},
            name='Неявная<br>схема'
        ),
            go.Scatter(
                x=np.array(range(len(T_implicit))) * h,
                y=T_explicit[0],
                mode='lines',
                line={'color': 'red'},
                name='Явная<br>схема'
            )
        ],
        layout=go.Layout(
            title='Распределение температуры по длине кристалла',
            xaxis={'title': 'Координата x, усл.ед.'},
            yaxis={'title': 'Температура T, усл.ед.'},
            updatemenus=[dict(
                type='buttons',
                buttons=[dict(label='Play',
                              method='animate',
                              args=[None, {'frame': {'duration': 50}, 'fromcurrent': True,
                                           'transition': {'duration': 0}}]),
                         dict(label='Pause',
                              method='animate',
                              args=[[None], {'frame': {'duration': 0}, 'redraw': False, 'mode': 'immediate',
                                             'transition': {'duration': 0}}])
                         ])]
        ),
        frames=[go.Frame(
            data=[go.Scatter(
                x=np.array(range(len(T_implicit))) * h,
                y=T_implicit[frame_num],
                mode='lines',
                line={'color': 'blue'},
                name='Неявная<br>схема'
            ),
                go.Scatter(
                    x=np.array(range(len(T_implicit))) * h,
                    y=T_explicit[frame_num],
                    mode='lines',
                    line={'color': 'red'},
                    name='Явная<br>схема'
                )
            ])
            for frame_num in range(len(T_implicit) - 1)]
    )

    fig2.show()

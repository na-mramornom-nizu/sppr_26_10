"""Build the energy-planning notebook from the model on pages 60–62."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
CELLS = []


def md(source: str) -> None:
    CELLS.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source: str) -> None:
    CELLS.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md(r"""
# Дополнительное задание: Оптимизация состава электростанций

**Дисциплина:** Системы поддержки принятия решений

**Студент:** Боронин Дмитрий Денисович

**Основа модели:** методическое пособие, § 3 «Применение линейного программирования к энергетике», с. 60–62, ограничения (3.2)–(3.6).

В методичке минимизируются годовые эксплуатационные расходы. Здесь по условию дополнительного задания максимизируется отношение годовой выработки к капитальным и эксплуатационным затратам за первый год. Это показатель выработки на рубль затрат, а не физический КПД электростанции.

Формулы (3.1)–(3.6) методички задают непрерывную модель $x_j\ge0$, но на с. 62 отдельно отмечено, что физическое число станций целое. Прямая и двойственная задачи ниже относятся к непрерывной модели. Целочисленный план рассчитывается отдельно и не участвует в равенстве прямого и двойственного оптимумов. Числовых коэффициентов в методичке нет; далее используется синтетический пример.
""")

md(r"""
## Исходные данные

Для типа станции $j$ заданы гарантированная мощность $a_j$ и пиковая мощность $b_j$ (МВт), годовая выработка $c_j$ (ГВт·ч/год), капитальные затраты $d_j$ (млн руб.) и годовые эксплуатационные расходы $f_j$ (млн руб./год). В знаменателе дроби $q_j=d_j+f_j$ означает затраты за первый год. Требования $A,B,C,D$ имеют соответствующие единицы.
""")

code('''
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    from IPython.display import display
    from pyomo.opt import TerminationCondition

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    stations = pd.DataFrame({
        "Тип": ["Базовая", "Пиковая", "Универсальная"],
        "a, МВт": [40, 90, 65],
        "b, МВт": [55, 150, 90],
        "c, ГВт·ч/год": [320, 330, 420],
        "d, млн руб.": [70, 160, 130],
        "f, млн руб./год": [10, 30, 20],
    }).set_index("Тип")
    stations["q = d + f, млн руб."] = (
        stations["d, млн руб."] + stations["f, млн руб./год"]
    )

    A, B, C, D = 235, 350, 1300, 430
    requirements = pd.DataFrame({
        "Показатель": ["Гарантированная мощность A", "Пиковая мощность B",
                       "Годовая выработка C", "Бюджет D"],
        "Требование": [A, B, C, D],
        "Единица": ["МВт", "МВт", "ГВт·ч", "млн руб."],
    })
    display(stations)
    display(requirements.set_index("Показатель"))
''')

md(r"""
## Математическая модель

$x_j\ge0$ — число станций типа $j$, $q_j=d_j+f_j>0$. В методичке целевая функция имеет вид $\min L(x)=\sum_j f_jx_j$; в дополнительном задании вместо неё рассматривается дробно-линейная задача:

$$
\max_{x\ge0}\quad
Z(x)=\frac{\sum_{j=1}^{3}c_jx_j}{\sum_{j=1}^{3}q_jx_j}
$$

$$
\sum_j a_jx_j\ge A,\qquad
\sum_j b_jx_j\ge B,\qquad
\sum_j c_jx_j\ge C,\qquad
\sum_j d_jx_j\le D.
$$

Подстановка Чарнеса—Купера:

$$
t=\left(\sum_j q_jx_j\right)^{-1},\qquad y_j=tx_j.
$$

После умножения ограничений на $t$ и нормировки знаменателя получается прямая задача линейного программирования:

$$
\begin{aligned}
\max_{y,t}\quad &\sum_j c_jy_j\\
\sum_j q_jy_j &=1,\\
\sum_j a_jy_j &\ge At,\\
\sum_j b_jy_j &\ge Bt,\\
\sum_j c_jy_j &\ge Ct,\\
\sum_j d_jy_j &\le Dt,\\
y_j,t&\ge0.
\end{aligned}
$$

Здесь $t>0$: из $\sum_j q_jy_j=1$ и $d_j>0$ следует $\sum_jd_jy_j>0$, а бюджет даёт $Dt\ge\sum_jd_jy_j$. Поэтому обратная замена $x_j=y_j/t$ определена и даёт допустимый план дробно-линейной задачи с непрерывными переменными.
""")

md(r"""
## Прямая задача в Pyomo
""")

code('''
    def solve(model):
        """Решить линейную модель и проверить статус HiGHS."""
        solver = pyo.SolverFactory("appsi_highs")
        if not solver.available():
            raise RuntimeError("Установите pyomo и highspy из requirements.txt")
        result = solver.solve(model)
        if result.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"HiGHS: {result.solver.termination_condition}")
        return model

    types = list(stations.index)
    a = stations["a, МВт"].to_dict()
    b = stations["b, МВт"].to_dict()
    c = stations["c, ГВт·ч/год"].to_dict()
    d = stations["d, млн руб."].to_dict()
    f = stations["f, млн руб./год"].to_dict()
    q = stations["q = d + f, млн руб."].to_dict()

    primal = pyo.ConcreteModel()
    primal.J = pyo.Set(initialize=types, ordered=True)
    # По Чарнесу—Куперу t — обратный знаменатель, y_j = t·x_j.
    primal.y = pyo.Var(primal.J, domain=pyo.NonNegativeReals)
    primal.t = pyo.Var(domain=pyo.NonNegativeReals)

    # Нормировка знаменателя делает дробную цель линейной.
    primal.efficiency = pyo.Objective(
        expr=sum(c[j] * primal.y[j] for j in primal.J),
        sense=pyo.maximize,
    )
    primal.normalization = pyo.Constraint(
        expr=sum(q[j] * primal.y[j] for j in primal.J) == 1
    )
    # Все ресурсные требования умножены на одну и ту же переменную t.
    primal.guaranteed = pyo.Constraint(
        expr=sum(a[j] * primal.y[j] for j in primal.J) >= A * primal.t
    )
    primal.peak = pyo.Constraint(
        expr=sum(b[j] * primal.y[j] for j in primal.J) >= B * primal.t
    )
    primal.generation = pyo.Constraint(
        expr=sum(c[j] * primal.y[j] for j in primal.J) >= C * primal.t
    )
    primal.budget = pyo.Constraint(
        expr=sum(d[j] * primal.y[j] for j in primal.J) <= D * primal.t
    )
    solve(primal)

    t_opt = pyo.value(primal.t)
    if t_opt <= 0:
        raise RuntimeError("Нулевая нормировка исключает обратную замену")
    y_opt = {j: pyo.value(primal.y[j]) for j in types}
    x_opt = {j: y_opt[j] / t_opt for j in types}
    total_generation = sum(c[j] * x_opt[j] for j in types)
    capital_cost = sum(d[j] * x_opt[j] for j in types)
    operating_cost = sum(f[j] * x_opt[j] for j in types)
    efficiency = total_generation / (capital_cost + operating_cost)

    plan = pd.DataFrame({
        "Тип станции": types,
        "y_j": [y_opt[j] for j in types],
        "x_j = y_j / t": [x_opt[j] for j in types],
    })
    indicators = pd.DataFrame({
        "Показатель": ["Гарантированная мощность", "Пиковая мощность",
                       "Годовая выработка", "Строительство",
                       "Эксплуатация за год", "Совокупные затраты"],
        "Значение": [sum(a[j] * x_opt[j] for j in types),
                     sum(b[j] * x_opt[j] for j in types),
                     total_generation, capital_cost, operating_cost,
                     capital_cost + operating_cost],
        "Требование": [f"≥ {A}", f"≥ {B}", f"≥ {C}",
                       f"≤ {D}", "—", "—"],
    })
    display(plan.round({"y_j": 6, "x_j = y_j / t": 3}).set_index("Тип станции"))
    print(f"t = {t_opt:.6f}")
    display(indicators.round(3).set_index("Показатель"))
    print(f"Эффективность Z = {efficiency:.3f} ГВт·ч/млн руб.")
    assert np.isclose(efficiency, pyo.value(primal.efficiency), atol=1e-9)
''')

md(r"""
## Двойственная задача

Для вывода двойственной модели запишем четыре неравенства прямой LP в форме «$\le0$»:

$$
At-\sum_j a_jy_j\le0,\quad
Bt-\sum_j b_jy_j\le0,\quad
Ct-\sum_j c_jy_j\le0,\quad
\sum_j d_jy_j-Dt\le0.
$$

Им соответствуют множители $\lambda_A,\lambda_B,\lambda_C,\lambda_D\ge0$. Равенству $\sum_jq_jy_j=1$ соответствует свободная переменная $u\in\mathbb R$. Двойственная задача:

$$
\begin{aligned}
\min\quad &u\\
q_ju-a_j\lambda_A-b_j\lambda_B-c_j\lambda_C
  +d_j\lambda_D&\ge c_j &&(j=1,2,3),\\
A\lambda_A+B\lambda_B+C\lambda_C-D\lambda_D&\ge0,\\
\lambda_A,\lambda_B,\lambda_C,\lambda_D&\ge0,\qquad u\in\mathbb R.
\end{aligned}
$$

Последнее неравенство отвечает столбцу неотрицательной переменной $t$. Так как в найденном плане $t>0$, оно выполняется как равенство по условию дополняющей нежёсткости.
""")

md(r"""
## Двойственная задача в Pyomo
""")

code('''
    dual = pyo.ConcreteModel()
    dual.J = pyo.Set(initialize=types, ordered=True)
    dual.resource = pyo.Set(initialize=["A", "B", "C", "D"])
    dual.u = pyo.Var(domain=pyo.Reals)
    dual.lam = pyo.Var(dual.resource, domain=pyo.NonNegativeReals)
    dual.bound = pyo.Objective(expr=dual.u, sense=pyo.minimize)

    def station_column(model, j):
        # Каждый столбец y_j даёт отдельное двойственное ограничение.
        return (
            q[j] * model.u
            - a[j] * model.lam["A"]
            - b[j] * model.lam["B"]
            - c[j] * model.lam["C"]
            + d[j] * model.lam["D"] >= c[j]
        )

    dual.station = pyo.Constraint(dual.J, rule=station_column)
    # Столбец t имеет нулевой коэффициент в прямой целевой функции.
    dual.scale = pyo.Constraint(
        expr=A * dual.lam["A"] + B * dual.lam["B"]
             + C * dual.lam["C"] - D * dual.lam["D"] >= 0
    )
    solve(dual)

    dual_values = pd.DataFrame({
        "Множитель": ["λ_A", "λ_B", "λ_C", "λ_D", "u (свободная)"],
        "Значение": [pyo.value(dual.lam[k]) for k in ("A", "B", "C", "D")]
                    + [pyo.value(dual.u)],
    })
    primal_value = pyo.value(primal.efficiency)
    dual_value = pyo.value(dual.bound)
    display(dual_values.round(3).set_index("Множитель"))
    print(f"Прямая цель: {primal_value:.6f}")
    print(f"Двойственная цель: {dual_value:.6f}")
    print(f"Разрыв двойственности: {abs(primal_value - dual_value):.9f}")
    print(f"Предельный эффект бюджета t·λ_D: "
          f"{t_opt * pyo.value(dual.lam['D']):.6f} "
          "ГВт·ч/млн руб. на 1 млн руб. бюджета")
    assert np.isclose(primal_value, dual_value, atol=1e-8)
''')

md(r"""
## Целочисленный план строительства

Для физически реализуемого плана добавляется условие $x_j\in\mathbb Z_{\ge0}$. Дробная целевая функция при этом уже не превращается в обычную LP посредством замены $y_j=tx_j$: целочисленность $x_j=y_j/t$ не является линейным ограничением. На трёх синтетических типах целочисленный оптимум можно найти методом Динкельбаха. При фиксированном $\eta$ решается задача

$$
\Phi(\eta)=\max_{x\in X_{\mathbb Z}}
\sum_{j=1}^{3}(c_j-\eta q_j)x_j,
\qquad q_j=d_j+f_j,
$$

где $X_{\mathbb Z}$ задаётся четырьмя исходными ограничениями и целочисленностью. Оптимальная эффективность $\eta^*$ удовлетворяет $\Phi(\eta^*)=0$. Двойственная LP даёт верхнюю оценку для этого плана, но равенство сильной двойственности относится только к непрерывной задаче.
""")

code('''
    def build_integer_model(eta):
        """Построить целочисленную задачу для текущей оценки эффективности."""
        model = pyo.ConcreteModel()
        model.J = pyo.Set(initialize=types, ordered=True)
        model.x = pyo.Var(model.J, domain=pyo.NonNegativeIntegers)
        model.score = pyo.Objective(
            expr=sum((c[j] - eta * q[j]) * model.x[j] for j in model.J),
            sense=pyo.maximize,
        )
        model.guaranteed = pyo.Constraint(
            expr=sum(a[j] * model.x[j] for j in model.J) >= A
        )
        model.peak = pyo.Constraint(
            expr=sum(b[j] * model.x[j] for j in model.J) >= B
        )
        model.generation = pyo.Constraint(
            expr=sum(c[j] * model.x[j] for j in model.J) >= C
        )
        model.budget = pyo.Constraint(
            expr=sum(d[j] * model.x[j] for j in model.J) <= D
        )
        return model


    eta = 0.0
    for iteration in range(1, 31):
        integer_model = solve(build_integer_model(eta))
        x_integer = {
            j: int(round(pyo.value(integer_model.x[j]))) for j in types
        }
        integer_generation = sum(c[j] * x_integer[j] for j in types)
        integer_cost = sum(q[j] * x_integer[j] for j in types)
        residual = integer_generation - eta * integer_cost
        integer_efficiency = integer_generation / integer_cost

        # Корень Phi(eta) = 0 даёт оптимум дробной целевой функции.
        if abs(residual) < 1e-8:
            break
        eta = integer_efficiency
    else:
        raise RuntimeError("Метод Динкельбаха не сошёлся")

    integer_plan = pd.DataFrame({
        "Тип станции": types,
        "Количество": [x_integer[j] for j in types],
    }).set_index("Тип станции")
    integer_checks = pd.DataFrame({
        "Показатель": [
            "Гарантированная мощность, МВт",
            "Пиковая мощность, МВт",
            "Годовая выработка, ГВт·ч",
            "Строительство, млн руб.",
            "Эксплуатация, млн руб./год",
        ],
        "Факт": [
            sum(a[j] * x_integer[j] for j in types),
            sum(b[j] * x_integer[j] for j in types),
            integer_generation,
            sum(d[j] * x_integer[j] for j in types),
            sum(f[j] * x_integer[j] for j in types),
        ],
        "Требование": [f"≥ {A}", f"≥ {B}", f"≥ {C}", f"≤ {D}", "—"],
    }).set_index("Показатель")
    display(integer_plan)
    display(integer_checks)
    print(f"Эффективность целочисленного плана: "
          f"{integer_efficiency:.3f} ГВт·ч/млн руб.")
    print(f"Верхняя оценка непрерывной LP: {primal_value:.3f}")

    assert abs(residual) < 1e-8
    assert integer_efficiency <= primal_value + 1e-8
''')

md(r"""
## Интерпретация решения

Оптимум непрерывной дробно-линейной модели: 5 базовых и 0,5 пиковой станции; универсальные станции не используются. Годовая выработка составляет 1765 ГВт·ч, капитальные затраты — 430 млн руб., эксплуатационные расходы за год — 65 млн руб. Эффективность за первый год равна $1765/495\approx3{,}566$ ГВт·ч на 1 млн руб. совокупных затрат. Это верхняя оценка для целочисленного строительства.

Пиковая мощность и строительный бюджет используются полностью. Поэтому $\lambda_B>0$ и $\lambda_D>0$; запас по гарантированной мощности и годовой выработке соответствует $\lambda_A=\lambda_C=0$. Величина $\lambda_D$ — оценка нормированного бюджетного ограничения. При малом увеличении исходного бюджета $D$ предельное изменение оптимальной эффективности равно $t\lambda_D\approx0{,}029$ единицы показателя на дополнительный 1 млн руб. бюджета, пока оптимальный базис не меняется.

Оптимальные значения прямой и двойственной LP совпадают: $Z^*=u^*\approx3{,}566$. Это численная проверка сильной двойственности.

Физически реализуемый целочисленный оптимум: 2 базовые, 1 пиковая и 1 универсальная станция. Для него выработка составляет 1390 ГВт·ч, строительные затраты — 430 млн руб., эксплуатационные расходы — 70 млн руб./год, эффективность — $1390/500=2{,}780$ ГВт·ч на 1 млн руб. затрат первого года. Значения $3{,}566$ и $2{,}780$ относятся к разным областям допустимых планов.
""")

NOTEBOOK = nbf.v4.new_notebook(cells=CELLS)
NOTEBOOK.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
NOTEBOOK.metadata["language_info"] = {"name": "python", "version": "3.12"}
target = ROOT / "sppr_bonus_energy.ipynb"
nbf.write(NOTEBOOK, target)
print(f"Created {target} ({len(CELLS)} cells)")

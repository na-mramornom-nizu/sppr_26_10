"""Build the standalone bonus notebook on power-plant composition."""

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

Количество станций рассматривается как вещественная переменная линейной модели. Численные параметры ниже синтетические; показатель эффективности имеет размерность ГВт·ч на 1 млн руб. совокупных затрат за рассматриваемый год.
""")

md(r"""
## Исходные данные

Для типа станции $j$ заданы гарантированная мощность $a_j$ и пиковая мощность $b_j$ (МВт), годовая выработка $c_j$ (ГВт·ч), капитальные затраты $d_j$ и годовые эксплуатационные расходы $f_j$ (млн руб.). Требования $A,B,C,D$ имеют соответствующие единицы.
""")

code('''
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
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
    print(stations.to_string())
    print("\\nТребования региона")
    print(requirements.to_string(index=False))
''')

md(r"""
## Математическая модель

$x_j\ge0$ — число станций типа $j$, $q_j=d_j+f_j>0$. Исходная дробно-линейная задача:

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

Здесь $t>0$: из $\sum_j q_jy_j=1$ и положительности $d_j$ следует $\sum_jd_jy_j>0$, а бюджет даёт $Dt\ge\sum_jd_jy_j$. Поэтому обратная замена $x_j=y_j/t$ определена.
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
    print(plan.round(3).to_string(index=False))
    print(f"t = {t_opt:.6f}")
    print("\\nПроверка исходных ограничений")
    print(indicators.round(3).to_string(index=False))
    print(f"Эффективность Z = {efficiency:.6f} ГВт·ч/млн руб.")
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
    print(dual_values.round(3).to_string(index=False))
    print(f"Прямая цель: {primal_value:.6f}")
    print(f"Двойственная цель: {dual_value:.6f}")
    print(f"Разрыв двойственности: {abs(primal_value - dual_value):.9f}")
    print(f"Предельный эффект бюджета t·λ_D: "
          f"{t_opt * pyo.value(dual.lam['D']):.6f} "
          "ГВт·ч/млн руб. на 1 млн руб. бюджета")
    assert np.isclose(primal_value, dual_value, atol=1e-8)
''')

md(r"""
## Интерпретация решения

Оптимальный состав в непрерывной модели: 5 базовых и 0,5 пиковой станции; универсальные станции не строятся. Годовая выработка составляет 1765 ГВт·ч, капитальные затраты — 430 млн руб., эксплуатационные — 65 млн руб. Эффективность равна $1765/495\approx3{,}566$ ГВт·ч на 1 млн руб. совокупных затрат. Дробное число станций допустимо только в принятой LP-постановке.

Пиковая мощность и строительный бюджет используются полностью. Поэтому $\lambda_B>0$ и $\lambda_D>0$; запас по гарантированной мощности и годовой выработке соответствует $\lambda_A=\lambda_C=0$. Величина $\lambda_D$ относится к нормированному бюджетному ограничению. При малом увеличении исходного бюджета $D$ предельное изменение оптимальной эффективности равно $t\lambda_D\approx0{,}029$ ГВт·ч/млн руб. на дополнительный 1 млн руб. бюджета, пока оптимальный базис не меняется.

Оптимальные значения прямой и двойственной LP совпадают: $Z^*=u^*\approx3{,}566$. Это численная проверка сильной двойственности.
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

"""Build the energy-planning notebook from section 3, pages 60–62."""

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

**Источник модели:** методическое пособие, § 3 «Применение линейного программирования к энергетике», с. 60–62, формулы (3.1)–(3.6).
""")

md(r"""
## Постановка и исходные данные

Для каждого типа станции $j=1,\ldots,n$ заданы гарантированная мощность $a_j$, пиковая мощность $b_j$, годовая выработка $c_j$, стоимость строительства $d_j$ и годовые эксплуатационные расходы $f_j$. Требуется выбрать состав станций с минимальными годовыми эксплуатационными расходами при выполнении требований $A$, $B$, $C$ и бюджета $D$.

В методичке приведена общая модель без числовой таблицы. Ниже — **синтетический пример** для трёх типов станций. Единицы мощности — МВт, выработки — ГВт·ч/год, затрат — млн руб.
""")

code('''
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    from IPython.display import display
    from pyomo.opt import TerminationCondition

    stations = pd.DataFrame({
        "Тип": ["Базовая", "Пиковая", "Универсальная"],
        "a, МВт": [40, 90, 65],
        "b, МВт": [55, 150, 90],
        "c, ГВт·ч/год": [320, 330, 420],
        "d, млн руб.": [70, 160, 130],
        "f, млн руб./год": [10, 30, 20],
    }).set_index("Тип")

    A, B, C, D = 235, 350, 1300, 430
    requirements = pd.DataFrame({
        "Показатель": [
            "Гарантированная мощность A",
            "Пиковая мощность B",
            "Годовая выработка C",
            "Бюджет строительства D",
        ],
        "Требование": [A, B, C, D],
        "Единица": ["МВт", "МВт", "ГВт·ч/год", "млн руб."],
    }).set_index("Показатель")

    display(stations)
    display(requirements)
''')

md(r"""
## Математическая модель по методичке

$x_j$ — число станций типа $j$. Целевая функция (3.1) и ограничения (3.2)–(3.6):

$$
\begin{aligned}
\min_{x}\quad L(x)&=\sum_{j=1}^{n}f_jx_j,\\
\sum_{j=1}^{n}a_jx_j&\ge A,\\
\sum_{j=1}^{n}b_jx_j&\ge B,\\
\sum_{j=1}^{n}c_jx_j&\ge C,\\
\sum_{j=1}^{n}d_jx_j&\le D,\\
x_j&\ge 0,\qquad j=1,\ldots,n.
\end{aligned}
$$

Затраты $d_j$ входят только в ограничение бюджета; целевая функция учитывает годовые эксплуатационные расходы $f_j$. В записи (3.1)–(3.6) переменные непрерывны. На с. 62 методички отмечено, что физически число станций должно быть целым и простое округление LP-решения может нарушить ограничения. Поэтому ниже решаются обе постановки: исходная линейная модель и её целочисленный вариант с $x_j\in\mathbb Z_{\ge0}$.
""")

md(r"""
## Решение в Pyomo
""")

code('''
    def solve(model):
        """Решить модель и проверить оптимальность результата HiGHS."""
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


    def build_model(integer=False):
        """Построить модель (3.1)–(3.6) с выбранной областью переменных."""
        model = pyo.ConcreteModel()
        model.J = pyo.Set(initialize=types, ordered=True)

        # Целочисленность добавляется только для практического плана.
        domain = pyo.NonNegativeIntegers if integer else pyo.NonNegativeReals
        model.x = pyo.Var(model.J, domain=domain)

        # В цели методички входят только годовые эксплуатационные расходы.
        model.cost = pyo.Objective(
            expr=sum(f[j] * model.x[j] for j in model.J),
            sense=pyo.minimize,
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


    lp = solve(build_model())
    mip = solve(build_model(integer=True))

    x_lp = {j: pyo.value(lp.x[j]) for j in types}
    x_mip = {j: int(round(pyo.value(mip.x[j]))) for j in types}

    plan = pd.DataFrame({
        "Тип станции": types,
        "LP, станций": [x_lp[j] for j in types],
        "Целочисленный план, станций": [x_mip[j] for j in types],
    }).set_index("Тип станции")
    display(plan.round(3))
''')

md(r"""
## Проверка ограничений и округления

Для каждого плана рассчитываются фактические мощности, выработка, строительные и эксплуатационные затраты. Округление LP-решения до ближайшего целого числа станций приведено только для проверки предупреждения на с. 62 методички; оно не является методом решения целочисленной задачи.
""")

code('''
    # Допуск устраняет погрешность решателя около 0,5 при округлении вверх.
    x_rounded = {j: int(np.floor(x_lp[j] + 0.5 + 1e-8)) for j in types}


    def totals(x):
        """Рассчитать показатели плана в единицах исходных данных."""
        return {
            "Гарантированная мощность, МВт": sum(a[j] * x[j] for j in types),
            "Пиковая мощность, МВт": sum(b[j] * x[j] for j in types),
            "Годовая выработка, ГВт·ч": sum(c[j] * x[j] for j in types),
            "Строительство, млн руб.": sum(d[j] * x[j] for j in types),
            "Эксплуатация, млн руб./год": sum(f[j] * x[j] for j in types),
        }


    check = pd.DataFrame({
        "LP": totals(x_lp),
        "Округление LP": totals(x_rounded),
        "Целочисленный оптимум": totals(x_mip),
    })
    check.insert(0, "Требование", [
        f"≥ {A}", f"≥ {B}", f"≥ {C}", f"≤ {D}", "минимум"
    ])
    display(check.round(3))

    lp_cost = pyo.value(lp.cost)
    mip_cost = pyo.value(mip.cost)
    print(f"Минимум LP: {lp_cost:.3f} млн руб./год")
    print(f"Минимум целочисленной модели: {mip_cost:.3f} млн руб./год")
    print(f"Разница: {mip_cost - lp_cost:.3f} млн руб./год")

    assert np.isclose(lp_cost, 65.0)
    assert np.isclose(mip_cost, 70.0)
    assert totals(x_mip)["Строительство, млн руб."] <= D
    assert totals(x_rounded)["Строительство, млн руб."] > D
''')

md(r"""
## Вывод

Решение непрерывной модели (3.1)–(3.6): 5 базовых и 0,5 пиковой станции; минимальные эксплуатационные расходы составляют 65 млн руб./год. Это нижняя оценка затрат для целочисленного плана, но половину станции построить нельзя.

Целочисленный оптимум: **2 базовые, 1 пиковая и 1 универсальная станция**. Он обеспечивает 235 МВт гарантированной мощности, 350 МВт пиковой мощности и 1390 ГВт·ч годовой выработки при строительном бюджете 430 млн руб. Минимальные годовые эксплуатационные расходы — **70 млн руб.**

Округление непрерывного решения до 5 базовых и 1 пиковой станции даёт строительные затраты 510 млн руб., то есть превышает бюджет на 80 млн руб. Разница между целочисленным оптимумом и нижней оценкой LP составляет 5 млн руб./год.
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

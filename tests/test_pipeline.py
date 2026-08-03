#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""СТАДИИ КОНВЕЙЕРА: распознавание, оракул, покрытие, реестр, отчёт.  Карта не нужна.

Каждая проверка здесь — про ЛОВУШКУ, а не про «работает ли функция».  Ловушки замерены и
описаны в docs/MEASURE_DISCIPLINE.md; тест не даёт им вернуться.

ЗАПУСК:  python3 tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, "%s: %s" % (type(e).__name__, e)))


# ---- РАСПОЗНАВАНИЕ --------------------------------------------------------------------------
def t_recognize_real_input():
    from tempo.core.op.recognize import from_file

    op = from_file(
        os.path.join(ROOT, "kernels", "naive", "gemm_fp16.cu"),
        {"M": 4096, "N": 15360, "K": 3840},
    )
    assert op.kind == "gemm" and op.dtype_acc == "fp32", op
    assert op.shapes["K"] == 3840


def t_recognize_refuses_unknown():
    from tempo.core.op.recognize import NotRecognised, parse_block

    try:
        parse_block("__global__ void mystery(float* x) {}")
    except NotRecognised:
        return
    raise AssertionError("незнакомое ядро НЕ получило отказа -- получит плохой код")


def t_foreign_layout_is_refused():
    """ОБЪЯВЛЕННАЯ ЧУЖАЯ РАСКЛАДКА -- ОТКАЗ, А НЕ ДРУГАЯ ОПЕРАЦИЯ.

    Блок TEMPO-OP объявляет раскладку, и объявить в нём можно ЛЮБУЮ; скелет же считает
    РОВНО ОДНУ (C = A*B^T, оба операнда k-мажорные).  Пока проверки не было, `layout: b=n`
    принималось МОЛЧА и порождалось ядро ДРУГОЙ операции -- ровно то, что README и
    docs/NOT_YET.md обещают не делать.  Сверка значений это НЕ ЛОВИТ: ядро считает свою
    неверную интерпретацию согласованно.
    """
    from tempo.plugins import registry
    from tempo.plugins.base import OpSpec, PluginCapabilityError

    p = registry.load("sm70")
    good = OpSpec(
        "gemm",
        "fp16",
        "fp16",
        "fp16",
        "fp32",
        "k",
        "k",
        "n",
        {"M": 128, "N": 4096, "K": 3840},
        tol_rel_l2=1e-3,
    )
    assert list(p.skeletons.variants(good)), "боевая раскладка перестала перечисляться"
    for bad, why in (
        (
            OpSpec(
                "gemm",
                "fp16",
                "fp16",
                "fp16",
                "fp32",
                "k",
                "n",
                "n",
                {"M": 128, "N": 4096, "K": 3840},
            ),
            "b=n",
        ),
        (
            OpSpec(
                "gemm",
                "fp16",
                "fp16",
                "fp16",
                "fp16",
                "k",
                "k",
                "n",
                {"M": 128, "N": 4096, "K": 3840},
            ),
            "накопитель fp16",
        ),
    ):
        try:
            list(p.skeletons.variants(bad))
        except PluginCapabilityError:
            continue
        raise AssertionError("%s принято МОЛЧА -- порождено ядро ДРУГОЙ операции" % why)


def t_recognize_catches_stale_description():
    """Описание, отставшее от кода, ловится ЗДЕСЬ, а не после часа замеров не того ядра."""
    from tempo.core.op.recognize import NotRecognised, check_signature, parse_block

    text = "// TEMPO-OP: gemm\n//   entry: was_renamed\n__global__ void actual_name(int a) {}"
    try:
        check_signature(parse_block(text), text)
    except NotRecognised:
        return
    raise AssertionError("сверка с сигнатурой пуска не сработала")


# ---- ПОКРЫТИЕ И ОРАКУЛ ----------------------------------------------------------------------
def t_coverage_catches_quarter_tile():
    """ГЛАВНАЯ ловушка: ядро считает ЧЕТВЕРТЬ плитки и ПРОХОДИТ сверку значений."""
    from tempo.core.op.coverage import check as cov

    stamps = [1] * 64 + [0] * 192  # посчитана четверть
    c = cov(stamps)
    assert not c.ok and c.zero == 192, c.render()


def t_coverage_catches_double_count():
    from tempo.core.op.coverage import check as cov

    c = cov([1] * 250 + [2] * 6)
    assert not c.ok and c.over == 6, c.render()


def t_oracle_gate_blocks_timing():
    from tempo.core.measure.correctness import must_pass_before_timing
    from tempo.core.op.oracle import gate

    res = gate([1.0, 2.0, 3.0], [1.0, 2.0, 9.0], tol=1e-6)
    try:
        must_pass_before_timing(res)
    except AssertionError:
        return
    raise AssertionError("секундомер не был запрещён при непройденном гейте")


def t_oracle_small_rell2_is_not_enough():
    """LAW=L-RELL2-NECESSARY-NOT-SUFFICIENT.  ПАДАЮЩАЯ ПРОВЕРКА ЗАКОНА.

    ЧТО ИМЕННО ОНА ЛОВИТ.  Встроенная сверка считает эталон НА ТЕХ ЖЕ собранных данных, поэтому
    ошибку СБОРКИ (раскладка, чужой вход, не тот кусок графа) она не видит: на дефекте 145 она
    показывала 2e-4 всё время, пока сетка несла мусор.  Значит идеальный relL2 НЕ ВПРАВЕ пускать
    секундомер сам по себе; пускает его только сверка с ЧУЖИМ путём, у которой НАЗВАНО, чем этот
    путь отличается.  Убрать закон -- значит вернуть `ok` к «значения сошлись», и тогда первый же
    из трёх случаев ниже пройдёт молча.
    """
    from tempo.core.measure.correctness import must_pass_before_timing
    from tempo.core.op.oracle import gate

    точно = [1.0, 2.0, 3.0]
    # 1. ЗНАЧЕНИЯ СОШЛИСЬ ТОЧНО, чужого пути НЕТ -- секундомер обязан остаться запрещён.
    res = gate(точно, точно, tol=1e-6)
    assert res.necessary_ok, "необходимое условие обязано выполняться: значения совпали"
    assert not res.ok, "нулевой relL2 БЕЗ чужого пути выдал разрешение на секундомер"
    try:
        must_pass_before_timing(res)
    except AssertionError as e:
        assert "НЕОБХОДИМОЕ, НЕ ДОСТАТОЧНОЕ" in str(e), str(e)
    else:
        raise AssertionError("секундомер разрешён по одной лишь сверке значений")

    # 2. ЧУЖОЙ ПУТЬ ПОСЧИТАН, НО НЕ НАЗВАН -- это вторая сверка того же самого.
    res2 = gate(точно, точно, tol=1e-6, independent=точно)
    assert not res2.ok, "безымянный «чужой путь» засчитан за ломку симметрии"
    assert "ЧУЖОЙ ПУТЬ НЕ НАЗВАН" in res2.render()

    # 3. НАЗВАН И СОШЁЛСЯ -- только теперь проход.
    res3 = gate(
        точно,
        точно,
        tol=1e-6,
        independent=точно,
        independent_path="счёт в двойной точности на разрежённой выборке ячеек",
    )
    assert res3.ok, res3.render()
    must_pass_before_timing(res3)


def t_oracle_speedup_needs_pair():
    """Метрика «против входа» ЧЕСТНА, но в одиночку -- самообман."""
    from tempo.core.measure.baseline import Baseline, Comparison, two_bars_required

    only_input = [
        Comparison(Baseline("INPUT", "наивный вход", "тот же компилятор"), 102.0)
    ]
    try:
        two_bars_required(only_input)
    except ValueError:
        pass
    else:
        raise AssertionError("одна планка прошла -- отчёт собрался бы самообманом")
    both = only_input + [
        Comparison(Baseline("LIBRARY", "библиотека", "то же умножение"), 0.94)
    ]
    two_bars_required(both)


# ---- ПРОИСХОЖДЕНИЕ --------------------------------------------------------------------------
def t_report_refuses_number_without_provenance():
    from tempo.core.report import Report
    from tempo.core.report.provenance import ProvenanceError
    from tempo.plugins.base import Rate

    r = Report(title="t", plugin_id="sm_70")
    r.rates_used = [Rate("X", 1.0, "ед", "MEASURED", prov=None)]
    try:
        r.render()
    except ProvenanceError:
        return
    raise AssertionError("ставка MEASURED без карты прошла в отчёт")


def t_spec_rate_forbidden_in_time_bound():
    from tempo.plugins import registry
    from tempo.plugins.base import ContractError, spec_forbidden_in_time_bound

    p = registry.load("sm70")
    try:
        spec_forbidden_in_time_bound(p.machine.peak("hbm_spec"))
    except ContractError:
        return
    raise AssertionError("паспортная ставка прошла в границу времени")


# ---- МОДЕЛЬ ---------------------------------------------------------------------------------
def t_model_is_allowed_to_be_silent():
    """Отсутствие связывающего канала -> МОЛЧАНИЕ, а не тихо неверная граница."""
    from tempo.core.model.bound import bound
    from tempo.plugins.base import Atom, AtomKind

    atoms = [Atom(0, AtomKind.COMPUTE, "X", {"НЕТ_ТАКОГО_КАНАЛА": 10.0}, 1.0)]
    b = bound(atoms, {})
    assert b.silent, "модель посчитала границу там, где ресурс не представлен"
    assert "МОДЕЛЬ МОЛЧИТ" in b.render()


def t_bank_law_is_parametric_in_bank_count():
    """Форма закона не знает числа банков -- оно ПАРАМЕТР."""
    from tempo.core.model.banks import degree

    assert degree(range(32), 1, banks=32) == 1.0
    assert degree(range(32), 32, banks=32) == 32.0
    assert degree(range(11), 7, banks=7) == 11.0  # подставная машина


def t_wave_quantum_not_round_numbers():
    """Полезные сетки кратны ЧИСЛУ ПРОЦЕССОРОВ, а не 64/128."""
    from tempo.core.model.wave import efficiency

    assert abs(efficiency(64, 80) - 0.8) < 1e-9, "квантизация волны посчитана неверно"
    assert abs(efficiency(80, 80) - 1.0) < 1e-9


# ---- РЕЕСТР ПОСТАВКИ ------------------------------------------------------------------------
def t_registry_always_has_fallback():
    from tempo.core.emit.registry import Entry, Registry

    r = Registry(
        op="gemm",
        arch="sm_70",
        entries=[Entry({"M_max": 16, "N_divides": 128}, "small")],
    )
    assert r.select(M=8, N=256) is not None
    assert r.select(M=8, N=100) is None, "непокрытая форма ДОЛЖНА уходить в откат"
    assert r.fallback, "у реестра нет отката -- поставка сломает чужой сервер"


def t_two_models_disagree_on_occupancy():
    """LAW=L-TWO-MODELS-OCCUPANCY.  В дереве ДВЕ модели каналов, и занятость входит в них
    ПО-РАЗНОМУ.  Это не дефект и не опровержение -- это НАЗВАННОЕ расхождение, которое иначе
    живёт молча и однажды даёт «тихо неверный вердикт».

    ЗДЕСЬ (конвейерная граница): канал ПРОЦЕССОРА домножается на число варпов, канал
    ПЛАНИРОВЩИКА -- нет, значит их ОТНОШЕНИЕ падает как 1/W, и связывающий канал с занятостью
    МЕНЯТЬСЯ МОЖЕТ.
    У СТЕНДА: обе нагрузки линейны по W (x W и x W/4), отношение ПОСТОЯННО, и замер это
    подтвердил -- прогон одного тела при 8/16/32 варпах дал одни и те же доли трижды.

    ПОЧЕМУ ЭТА ПРОВЕРКА ВАЖНА.  Опровержение LAW=L-OCCUPANCY-MOVES-BINDING («связывающий
    ресурс меняется с занятостью» -- неверно) ПРОВЕРЕНО ТОЛЬКО НА СТЕНДЕ.  Перенести его сюда
    без проверки значило бы повторить ровно ту ошибку, которую оно само описывает: величина
    верна, ОБЛАСТЬ не та.  Поэтому здесь стоит не перенос вывода, а СВОЯ проверка своей модели.
    """
    from tempo.core.model.bound import bound
    from tempo.plugins import registry
    from tempo.plugins.base import OpSpec

    p = registry.load("sm70")
    op = OpSpec(
        "gemm",
        "fp16",
        "fp16",
        "fp16",
        "fp32",
        "k",
        "k",
        "n",
        {"M": 4096, "N": 15360, "K": 3840},
        tol_rel_l2=1e-3,
    )
    h = list(p.skeletons.variants(op))[0]
    atoms = p.skeletons.estimate_atoms(op, h)
    ratios = []
    for w in (8, 16, 32):
        b = bound(atoms, p.machine.channels(), warps_per_sm=w)
        ratios.append(b.per_channel["ISSUE"] / b.per_channel["MIO"])
    assert all(r > 0 for r in ratios), (
        "каналы не участвуют в выводе -- проверять нечего"
    )
    for a, b_ in zip(ratios, ratios[1:]):
        assert abs(a / b_ - 2.0) < 1e-6, (
            "отношение канала планировщика к каналу процессора обязано падать РОВНО вдвое на "
            "удвоение занятости (получено %.4f). Если оно постоянно -- значит арифметика двух "
            "моделей совпала, и запись L-TWO-MODELS-OCCUPANCY надо снимать как мнимую."
            % (a / b_)
        )


CASES = [
    ("распознаётся боевой наивный вход", t_recognize_real_input),
    (
        "ДВЕ модели каналов расходятся по занятости, и это НАЗВАНО",
        t_two_models_disagree_on_occupancy,
    ),
    ("незнакомое ядро получает ОТКАЗ", t_recognize_refuses_unknown),
    (
        "чужая РАСКЛАДКА получает ОТКАЗ, а не другую операцию",
        t_foreign_layout_is_refused,
    ),
    ("описание, отставшее от кода, ловится", t_recognize_catches_stale_description),
    ("покрытие ловит ЧЕТВЕРТЬ плитки", t_coverage_catches_quarter_tile),
    ("покрытие ловит двойной счёт", t_coverage_catches_double_count),
    ("гейт корректности ЗАПРЕЩАЕТ секундомер", t_oracle_gate_blocks_timing),
    (
        "малый relL2 -- НЕОБХОДИМОЕ условие, не достаточное",
        t_oracle_small_rell2_is_not_enough,
    ),
    ("одна планка не проходит: нужны ДВЕ", t_oracle_speedup_needs_pair),
    (
        "число без происхождения не попадает в отчёт",
        t_report_refuses_number_without_provenance,
    ),
    (
        "паспортная ставка запрещена в границе времени",
        t_spec_rate_forbidden_in_time_bound,
    ),
    ("модель умеет МОЛЧАТЬ", t_model_is_allowed_to_be_silent),
    (
        "закон банков параметричен по числу банков",
        t_bank_law_is_parametric_in_bank_count,
    ),
    ("волна кратна процессорам, а не круглым числам", t_wave_quantum_not_round_numbers),
    ("у реестра поставки ВСЕГДА есть откат", t_registry_always_has_fallback),
]


def main():
    print("СТАДИИ КОНВЕЙЕРА (карта не нужна)")
    for name, fn in CASES:
        check(name, fn)
    ok = sum(1 for _, good, _ in RESULTS if good)
    for name, good, err in RESULTS:
        print(
            "  %-8s %s%s"
            % (
                "ПРОЙДЕН" if good else "ПАДЁТ",
                name,
                "" if good else "\n           " + err,
            )
        )
    print("ИТОГ: %d/%d" % (ok, len(RESULTS)))
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

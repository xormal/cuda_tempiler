#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПЛАГИН Volta / sm_70 (Tesla V100).  БОЕВОЙ: почти все ставки -- ЗАМЕР, не паспорт.

ЧТО ЗДЕСЬ ЛЕЖИТ И ПОЧЕМУ ИМЕННО ЗДЕСЬ
    data/machine/*.json  ставки с картой, частотами и числом чужих процессов;
    machine/memory/tensor/resources/sync/space/lower/toolchain/counters/banks/occupancy
                         восемь разделов контракта;
    isa_sass.py          разборщик SASS ЦЕЛИКОМ (он архитектурный по существу);
    skeletons/           100 % архитектурный код -- через границу НЕ ХОДИТ вовсе;
    gemm_bound.py        отсекатель без сборки: ЗАКОН из карты фрагмента + СТАВКИ раздельно.

ЧТО ЧЕРЕЗ ГРАНИЦУ НЕ ХОДИТ, кроме скелетов: ВЫВОДЫ.  «Занятость -- не рычаг», «kBJ=32 мёртв»,
«int8 не даёт ФЛОПов на sm_70» -- всё это выводы из НАШИХ ТЕЛ на ЭТОЙ машине.  Они живут в
notes()/docs плагина и в отчёте, а не в core/search/prune.py.  Правило «предпочитать низкую
занятость» внутри core было бы протечкой, и её ловят гейты G6+G8.

ЗАПУСК САМОПРОВЕРКИ:  python3 -m tempo.plugins.sm70
"""

from __future__ import annotations

from ..base import CONTRACT, Report, acc_regs_per_thread, operand_loads_per_mma
from .classifier import Sm70Classifier
from .counters import Sm70Meters
from .lower import Sm70Skeletons
from .machine import Sm70Machine
from .memory import Sm70Memory
from .resources import Sm70Resources
from .sync import Sm70Sync
from .tensor import Sm70TensorUnit
from .toolchain import Sm70Toolchain


class Sm70Plugin:
    id = "sm_70"
    contract = CONTRACT
    description = "Volta / Tesla V100-SXM2-32GB. Ставки сняты на карте 1 при 1530 МГц, 0 чужих процессов."

    def __init__(self):
        self.machine = Sm70Machine()
        self.memory = Sm70Memory()
        self.tensor = Sm70TensorUnit()
        self.resources = Sm70Resources()
        self.sync = Sm70Sync()
        self.classifier = Sm70Classifier()
        self.skeletons = Sm70Skeletons()
        self.toolchain = Sm70Toolchain()
        self.meters = Sm70Meters()

    # -- ЧЕСТНЫЙ ПЕРЕЧЕНЬ НЕРЕАЛИЗОВАННОГО ---------------------------------------------------
    def declared_stubs(self):
        return (
            "skeletons/fmha_fwd -- скелета внимания НЕТ; сравнивать с отгруженным ядром пока нечем",
            "MIO_CONFLICT -- ставка NOT_MEASURED: калибровочная поверхность дополнений ПУСТА "
            "(padsweep ни разу не доведён, замеренных кривых 0)",
            "LATENCY.LDG = 400 -- МОДЕЛЬ, стендом не снята, а правит всю глобальную часть",
            "PEAK.L2 = 2155 ГБ/с -- ОЦЕНКА, не замер",
            "канала «задержка при низкой занятости» в модели НЕТ, а при M<=64 связывает именно он",
            "classifier.encode_control -- не реализован намеренно (нет верификатора зависимостей T3)",
            "W8A8 (байтовые ОБА операнда) -- не построен: нужен байтовый операнд A и "
            "ДВУСТОРОННЯЯ свёртка смещения",
            "эпилогов нет (слияние silu(gate)*up: цена названа ~6 % одного матмуля)",
        )

    def notes(self):
        """ВЫВОДЫ. Они НЕ ходят через границу и НЕ участвуют в отсечении."""
        return self.resources.notes() + (
            "int8 на sm_70 НЕ ДАЁТ СЧЁТНОГО ВЫИГРЫША: IMMA нет, и int8 в мантиссе fp16 идёт ТОЙ ЖЕ "
            "инструкцией HMMA.884. Замеренное отношение к fp16 -- 1.00 на K=256..16384. "
            "Байтовый вес строится ради ТРАФИКА при малом M.",
            "kBJ=32 собирается (стена cutlass снята), но замеренно МЁРТВ: резидентность стоит "
            "32*d/kBI и от kBJ не зависит.",
            "Triton на Volta НЕ ПОРОЖДАЕТ тензорных инструкций вовсе (mma.sync 0, wmma 0, hmma 0), "
            "потолок 3.5 ТФЛОП/с. TileLang HMMA.884 эмитит, но 0.68x cuBLAS -- годен как ИЗМЕРИТЕЛЬ.",
            "Постоянная память при индексе от threadIdx СЕРИАЛИЗУЕТСЯ: обвал в 16 раз, таблица "
            "разворота формата -- в 56 раз. Никаких LUT в распаковке.",
        )

    # -- САМОПРОВЕРКА ------------------------------------------------------------------------
    def selftest(self) -> Report:
        r = Report("sm_70")
        M, T, R = self.machine, self.tensor, self.resources

        # 1. Ставки и их происхождение
        syms = M.symbols()
        r.check("таблица ставок непуста", len(syms) > 0, "%d символов" % len(syms))
        bad = [
            s.symbol
            for s in syms.values()
            if s.status == "MEASURED"
            and (
                s.prov is None or s.prov.card is None or s.prov.card.foreign_procs != 0
            )
        ]
        r.check(
            "каждая MEASURED несёт карту и 0 чужих процессов",
            not bad,
            ", ".join(bad[:3]),
        )
        r.check(
            "SPEC-паспорт HBM объявлен и ОТЛИЧАЕТСЯ от замера",
            M.peak("hbm_spec").value == 900.0 and M.peak("hbm_copy").value == 819.0,
            "900 паспорт против 819 замер",
        )

        # 1-бис. ТРИ РЕЖИМА ПОЛОСЫ, а не одно число (LAW=L-BW-THREE-MODES, задача 118).
        # Лечение расхождения 878/841 -- НЕ замена одного другим, а ТРЕТИЙ символ со своим
        # режимом: доля полосы, у которой не назван режим, ошибается на 4.4 % молча.
        modes = ("hbm_read_pure", "hbm_read", "hbm_copy")
        vals = [M.peak(k).value for k in modes]
        r.check(
            "полоса объявлена ТРЕМЯ режимами, а не одним числом",
            vals == [878.0, 841.0, 819.0],
            "чтение/сумма/копия = %s" % vals,
        )
        r.check("режимы РАЗЛИЧНЫ (иначе символы лишние)", len(set(vals)) == 3)
        r.check(
            "у КАЖДОГО режима в записи назван ОБРАЗЕЦ ДОСТУПА",
            all("РЕЖИМ:" in M.peak(k).note for k in modes),
            "число полосы без режима -- не число",
        )
        r.check(
            "разница чистого чтения и суммирования = 4.4 %",
            abs((878.0 / 841.0 - 1.0) * 100 - 4.4) < 0.05,
            "%.2f %% входит во ВСЕ доли, посчитанные против 841"
            % ((878.0 / 841.0 - 1) * 100),
        )

        # 2. Закрытая таблица отказывает, а не подставляет умолчание (гейт G4)
        from ..base import UnknownSymbol

        try:
            M.rate("CAP.NONEXISTENT")
            r.check("закрытая таблица отказывает на неизвестном символе", False)
        except UnknownSymbol:
            r.check("закрытая таблица отказывает на неизвестном символе", True)

        # 3. Регистровые законы -- ПРЯМОЙ ОПРОС ptxas дал ровно это (4/4)
        r.check("Q(12)=168", R.reg_budget(12) == 168)
        r.check("Q(16)=128", R.reg_budget(16) == 128)
        r.check("Q(24)=80", R.reg_budget(24) == 80)
        r.check("Q(32)=64", R.reg_budget(32) == 64)
        r.check("порог разлива = MaxLive+7", R.spill_threshold(100) == 107)
        r.check("первые два разлитых бесплатны", R.free_spills() == 2)

        # 4. Волна -- 80 SM, а не 64/128
        occ = R.occupancy(regs=128, smem_bytes=32768, threads=256)
        r.check(
            "волна кратна числу SM (80)",
            R.wave_quantum(occ) % 80 == 0,
            "квант = %d" % R.wave_quantum(occ),
        )

        # 5. Закон MIO: три предела
        wf_bcast = self.memory.wavefronts([0] * 32, width_bytes=16)
        r.check(
            "LDS.128 при ПОЛНОЙ рассылке = 2 вайвфронта (пол ширина/8)",
            abs(wf_bcast.wavefronts - 2.0) < 1e-9,
            "ncu: 2.000",
        )
        wf_stride32 = self.memory.wavefronts([l * 8 for l in range(32)], width_bytes=4)
        r.check(
            "шаг 32 Б, LDS.32 = 8 вайвфронтов",
            abs(wf_stride32.wavefronts - 8.0) < 1e-9,
            "ncu: 8.000",
        )
        wf_pad = self.memory.wavefronts([l * 33 for l in range(32)], width_bytes=4)
        r.check(
            "шаг 132 Б (дополнение на слово) = 1 вайвфронт",
            abs(wf_pad.wavefronts - 1.0) < 1e-9,
            "ncu: 1.000",
        )

        # 6. Тензорный узел: закон плитки ВЫВОДИТСЯ из карты фрагмента
        op = T.select(("fp16", "fp16"), "fp32")
        r.check(
            "накопитель плитки 16x16 = 8 float/поток (выведено из карты)",
            acc_regs_per_thread((16, 16), op) == 8,
        )
        r.check(
            "накопитель плитки 64x64 = 128 float/поток",
            acc_regs_per_thread((64, 64), op) == 128,
        )
        r.check(
            "загрузок-на-mma при 32x32 = 1.00",
            abs(operand_loads_per_mma((32, 32), op) - 1.0) < 1e-9,
        )
        r.check(
            "загрузок-на-mma при 64x64 = 0.50",
            abs(operand_loads_per_mma((64, 64), op) - 0.5) < 1e-9,
        )
        r.check("домен точности объявлен", op.exact_while == "|sum| < 2**24")
        from .tensor import a_value_fanout, bijection_ok

        r.check(
            "карта накопителя -- БИЕКЦИЯ на 16x16 (покрытие, а не значения)",
            bijection_ok(),
        )
        r.check(
            "значение A попадает РОВНО в две полосы (трафик Q неустраним)",
            a_value_fanout() == 2,
        )
        r.check(
            "две единицы команды НАЗВАНЫ порознь: квадропара 2.00, варповая 8.00",
            abs(M.rate("CAP.TENSOR").value - 2.0) < 1e-9
            and abs(op.cost.value - 8.0) < 1e-9,
        )

        # 7. Отсутствующее у Volta отказывает СТРУКТУРНО, а не падает
        from ..base import PluginCapabilityError

        try:
            T.select(("int8", "int8"), "int32")
            r.check("IMMA (int8+int8->int32) отвергается структурно", False)
        except PluginCapabilityError:
            r.check("IMMA (int8+int8->int32) отвергается структурно", True, "sm_75+")
        try:
            self.sync.rendezvous_cost("mbarrier", 8)
            r.check("mbarrier отвергается структурно", False)
        except PluginCapabilityError:
            r.check("mbarrier отвергается структурно", True, "sm_80+")

        # 8. Волтовское допущение объявлено ДАННЫМИ (иначе G8 нечего фальсифицировать)
        tx = self.sync.transactions()[0]
        r.check(
            "подача gmem->smem объявлена съедающей регистры",
            tx.consumes_registers is True,
        )
        r.check("ожидание объявлено НЕЯВНЫМ (табло, cp.async нет)", tx.wait_op is None)

        # 9. Скелеты и оси
        sk = self.skeletons
        r.check("скелеты объявляют операции", "gemm" in sk.ops())
        r.check("оси -- открытый словарь, непустой", len(sk.axes()) >= 8)

        # 10. Вердикт БЕЗ СБОРКИ работает и различает состояния
        v_fits = R.verdict(regs=128, max_live=100, smem_bytes=32768, threads=256)
        v_wall = R.verdict(regs=255, max_live=300, smem_bytes=32768, threads=256)
        v_smem = R.verdict(regs=128, max_live=100, smem_bytes=200000, threads=256)
        r.check("вердикт FITS", v_fits.code == "FITS", v_fits.explain)
        r.check("вердикт WALL_REG", v_wall.code == "WALL_REG", v_wall.explain)
        r.check("вердикт WALL_SMEM", v_smem.code == "WALL_SMEM", v_smem.explain)

        # 11-бис. ЗАКОНЫ ФОРМЫ: что связывает при данном M и до скольких бит сжимать вес.
        # LAW=L-MRIDGE-128, LAW=L-BIT-FLOOR, LAW=L-NARROW-0644.  Каждая проверка ПАДАЕТ, если
        # закон убрать: это и есть разница между законом и комментарием.
        from . import shape_law as SL
        from ..base import PluginCapabilityError as _PCE

        v32 = SL.binding("gate,up", 32)
        v4096 = SL.binding("gate,up", 4096)
        r.check(
            "при M=32 у gate,up связывает ПОЛОСА (86.0 % против 24.5 % счёта)",
            v32.kind == "ПОЛОСА" and abs(v32.bw - 0.860) < 1e-9,
            v32.explain[:96],
        )
        r.check(
            "при M=4096 у той же формы связывает СЧЁТ (96.1 % против 6.1 %)",
            v4096.kind == "СЧЁТ" and abs(v4096.comp - 0.961) < 1e-9,
            v4096.explain[:96],
        )
        # ПЕРЕЛОМ -- СВОЙСТВО НАБОРА ФОРМ, А НЕ ОТДЕЛЬНОЙ: у k,v доли сравнялись уже при 128.
        r.check("перелом объявлен на M=128", SL.ridge_m() == 128)
        r.check(
            "у k,v при M=128 модель МОЛЧИТ (40.4 против 42.4 -- внутри разрешения)",
            SL.binding("k,v", 128).kind == "МОЛЧИТ",
            "перелом есть свойство НАБОРА форм; на отдельной он сдвинут",
        )

        own, cal, crude, _e = SL.bit_floor("gate,up", 64, chain_ops=0.0)
        r.check(
            "дно gate,up при M=64 -- 8.03 бита (int8 УЖЕ на дне)",
            abs(own - 8.03) < 1e-9,
        )
        r.check(
            "дно k,v при M=64 вдвое ниже -- своя строка, а не строка gate,up",
            abs(SL.bit_floor("k,v", 64)[0] - 4.60) < 1e-9,
            "для ОДНОЙ матрицы берётся ЕЁ строка",
        )
        r.check(
            "ДВА наклона дна названы ОБА и расходятся на 16 %",
            abs(cal - 0.1236 * 64) < 1e-9
            and abs(crude - 0.144 * 64) < 1e-9
            and abs((crude / cal - 1) * 100 - 16.4) < 0.2,
            "модель выдачи %.2f против грубой крыши %.2f бита" % (cal, crude),
        )

        ok32, ratio32, _ = SL.narrow_ok("gate,up", 32)
        ok4096, ratio4096, _ = SL.narrow_ok("gate,up", 4096)
        r.check(
            "узкий формат ПРИМЕНИМ там, где связывает полоса (порог 0.644)",
            ok32 is True and ratio32 < 0.644,
            "счёт/полоса = %.3f" % ratio32,
        )
        r.check(
            "узкий формат ПЛАТИТ ВРЕМЕНЕМ там, где связывает счёт",
            ok4096 is False and ratio4096 > 0.644,
            "счёт/полоса = %.2f" % ratio4096,
        )
        r.check(
            "на форме, где модель молчит, вердикта о формате НЕТ",
            SL.narrow_ok("k,v", 128)[0] is None,
            "отказ, а не осторожность: неназванный связывающий ресурс = нет ответа",
        )

        # ОБЛАСТЬ ДЕЙСТВИЯ -- ЧАСТЬ ЗАКОНА.  Оба нарушения обязаны быть ОТКАЗОМ, а не числом.
        for what, fn in (
            ("незамеренная форма", lambda: SL.binding("mlp_gigantic", 32)),
            ("M вне размаха замеров", lambda: SL.binding("gate,up", 8)),
        ):
            try:
                fn()
                r.check(
                    "ОБЛАСТЬ: %s -> отказ" % what, False, "выдано число вместо отказа"
                )
            except _PCE:
                r.check("ОБЛАСТЬ: %s -> отказ" % what, True)
        v_i = SL.binding("gate,up", 48)
        r.check(
            "между замеренными точками ИНТЕРПОЛЯЦИЯ названа вслух",
            (not v_i.exact) and "ИНТЕРПОЛИРОВАНЫ" in v_i.explain,
        )

        # 11-трис. СТАВКА SFU НЕ ПЕРЕНОСИТСЯ НА КОНВЕРСИИ (LAW=L-SFU-RATE-IS-MUFU).
        # Проверка ПАДАЕТ, если кто-нибудь отнесёт конверсию к дефицитному каналу: это ровно
        # тот перенос замеренной величины с одного объекта на другой, который уже стоил
        # предсказания >= +50 % там, где замерено +32.3 %.
        C = self.classifier
        r.check(
            "трансцендентная идёт по дефицитному каналу со ставкой 8.00",
            C.classify("MUFU.EX2 R0, R1 ;").channel == "SFU"
            and abs(C.classify("MUFU.EX2 R0, R1 ;").cycles - 8.0) < 1e-9,
        )
        conv = [
            C.classify("%s.F32.S32 R0, R1 ;" % op)
            for op in ("I2F", "F2F", "F2I", "I2I")
        ]
        r.check(
            "КОНВЕРСИИ по дефицитному каналу НЕ идут (перенос ставки отвергнут)",
            all(c.channel != "SFU" for c in conv),
            "иначе модель нарушила бы собственный замер +32.3 %",
        )
        r.check(
            "у конверсий та же ставка, что во второй модели дерева (класс плавающих)",
            all(abs(c.cycles - 2.0) < 1e-9 for c in conv),
            "прежде они падали в умолчание 1.00 -- вдвое дешевле, чем у стенда",
        )
        # КАНАЛ -- КЛЮЧ ТАБЛИЦЫ КАНАЛОВ, А НЕ КЛАСС РАЗБОРЩИКА. Прежняя редакция клала сюда
        # класс («плавающие», «тензорные»), которого в таблице НЕТ ни одного: потребитель
        # получил бы промах и списал бы команду в никуда -- молча, потому что `classify` пока
        # никем не вызывается. Проверка ловит ЛАТЕНТНЫЙ тихо неверный вердикт.
        probes = (
            "HMMA.884.F32.F32 R0, R1, R2, R3 ;",
            "LDS.128 R0, [R1] ;",
            "MUFU.EX2 R0, R1 ;",
            "I2F.F32.S32 R0, R1 ;",
            "FFMA R0, R1, R2, R3 ;",
        )
        chans = {C.classify(t).channel for t in probes}
        r.check(
            "канал классификатора -- КЛЮЧ таблицы каналов, а не класс разборщика",
            chans <= set(M.channels()),
            "получено %s" % sorted(chans),
        )

        # 11. Честность
        r.check(
            "перечень заглушек непуст (иначе он врёт)", len(self.declared_stubs()) >= 5
        )
        return r


_P = None


def load():
    global _P
    if _P is None:
        _P = Sm70Plugin()
    return _P


if __name__ == "__main__":
    import sys

    rep = load().selftest()
    print(rep.render())
    for s in load().declared_stubs():
        print("  ЗАГЛУШКА: " + s)
    sys.exit(0 if rep.green else 1)

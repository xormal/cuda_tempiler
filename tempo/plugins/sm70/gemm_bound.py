# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ОТСЕКАТЕЛЬ БЕЗ СБОРКИ для плотного умножения на тензорных ядрах.

РАЗДЕЛЕНИЕ, КОТОРОЕ ЗДЕСЬ И ЕСТЬ ПРОДУКТ:
  * ЗАКОН (эта половина файла) выведен из КАРТЫ ФРАГМЕНТА и не содержит ни одного числа железа.
    Он одинаков для m8n8k4 и m16n8k16 -- меняются только размеры фрагмента и ставки.
  * СТАВКИ (`Rates`) -- данные плагина. Каждая несёт статус происхождения. Ни одна не вшита
    в закон; подмена ставки обязана двигать вывод (гейт возмущения).

ЧТО ЗАКОН УТВЕРЖДАЕТ:
  вайвфронтов разделяемой на ТАКТ ТЕНЗОРНОГО КОНВЕЙЕРА
      nu_mio = CONF * (MB+NB)/(2*MB*NB)   [чтение фрагментов]
             + 4/BM + 4/BN                [запись плитки]
  Канал MIO отдаёт 1 вайвфронт/такт/SM, тензорный -- 2 КВАДРОПАРЫ/такт/SM.
  nu_mio >= 1 означает: разделяемая связывает РАНЬШЕ тензорных ядер.

ЕДИНИЦА, НА КОТОРОЙ ЭТОТ ЗАКОН БЫЛ СНАЧАЛА НЕВЕРЕН В 4 РАЗА (записываю, потому что это ровно
та ошибка, от которой памятка предупреждает). Первая редакция считала HMMA.884 ОДНОЙ командой на
варп и брала 2 такта на неё. На самом деле mma.sync.m8n8k4 распадается на ЧЕТЫРЕ квадропарных
HMMA, и 2 такта -- цена КВАДРОПАРЫ. Отсюда закон давал 1.25 (разделяемая связывает) вместо
0.3125 (не связывает), и вывод "рычаг -- большая плитка варпа" был бы ложным.
Опровергнуто счётчиком: sm__inst_executed_pipe_tensor = 471 859 200 при M*N*K = 1.208e11, то есть
256 умножений-сложений на счётную единицу = ровно одна квадропара 8x8x4.
СЛЕДСТВИЕ ДЛЯ ПРОДУКТА: ставка "тактов на команду" бессмысленна без имени ЕДИНИЦЫ команды.

CONF -- множитель конфликтности чтения фрагментов. Он НЕ константа железа, а функция РАСКЛАДКИ:
  SWZ=1 (исправленный свиззл)  -> 1.00 (пол: 64 различных слова на 32 банка = 2 вайвфронта)
  SWZ=0 (свиззл как отгружен)  -> 3.02 ЗАМЕРЕНО (178.0 млн против 59.0 млн расчётных)
"""

# ---------------------------------------------------------------- СТАВКИ ПЛАГИНА sm_70
# Каждая строка: (значение, статус, чем подтверждено).
RATES = {
    # тензорная инструкция HMMA.884 = одна квадропара 8x8x4 = 256 умножений-сложений
    "tensor_qp_per_cycle_sm": (
        2.0,
        "MEASURED",
        "125.3 ТФЛОП/с @1530 / 80 SM / 256 МАС; стенд 1.95 HMMA/такт/SM",
    ),
    "mio_wavefronts_per_cycle_sm": (
        1.0,
        "MEASURED",
        "probe.cu, 6 точек, <1%; ncu 17/17",
    ),
    "regfile_words_per_sm": (65536, "SPEC", "паспорт V100"),
    "reg_isa_limit": (255, "SPEC", "потолок ISA sm_70"),
    "smem_per_sm": (96 * 1024, "SPEC", "паспорт"),
    "warp_slots_per_sm": (64, "SPEC", "паспорт"),
    "sms": (80, "SPEC", "паспорт V100-SXM2"),
    "reg_granularity_per_warp": (
        256,
        "MEASURED",
        "Q(W)=min(255,8*floor(256/W)), опрос ptxas 4/4",
    ),
    # ПОЛОСА ОШИБКИ СОБСТВЕННОЙ ОЦЕНКИ РЕГИСТРОВ -- ЗАМЕРЕНА против ptxas на 12 точках, где
    # есть И оценка, И сборка (kernels/shipped/gemm_fp16/sm_70/manifest.json, раздел "build").
    # Оценка занижает до 45.1 % и завышает до 15.3 %. Для ОТСЕЧЕНИЯ важен только верхний край:
    # завышение -- это и есть выброшенный победитель.
    "reg_estimate_over_max": (
        0.153,
        "MEASURED",
        "manifest.json.build, 12 точек: -45.1 %..+15.3 % против ptxas; "
        "худшее завышение +15.3 % (m256x128k32: оценка 272, ptxas 236)",
    ),
}
# Множитель конфликтности чтения фрагментов при ОТГРУЖЕННОМ свиззле (нижние разряды строки).
CONF_LEGACY = (
    3.02,
    "MEASURED",
    "ncu wf_ld 177.97e6 против расчётных 58.98e6, 128x128x32, 4 варпа",
)
# ФОРМА ФРАГМЕНТА (данные плагина -- это карта mma.sync.m8n8k4 на Volta)
FRAG = {
    "tile_m": 16,
    "tile_n": 16,
    "k_step": 4,  # плотная плитка одного mma на варп
    "acc_regs_per_tile": 8,  # накопителей на полосу на плитку 16x16
    "frag_bytes_per_lane_per_kstep": 8,  # 4 half подряд = один LDS.64; пара шагов = LDS.128
}


def q_budget(warps_per_sm):
    """Регистровый бюджет на нить при заданной занятости. Замерено: совпадает с ptxas 4/4."""
    return min(
        RATES["reg_isa_limit"][0],
        8
        * (RATES["reg_granularity_per_warp"][0] // (32 * warps_per_sm) * 32 // 32 * 1),
    )


def occupancy(regs, smem_bytes, threads_per_cta):
    """Занятость: варпов на SM и блоков на SM. Форма закона -- конвейерная, числа -- плагинные."""
    nw = threads_per_cta // 32
    by_reg = RATES["regfile_words_per_sm"][0] // (regs * 32) if regs else 64
    by_reg = min(by_reg, RATES["warp_slots_per_sm"][0])
    ctas_reg = by_reg // nw
    ctas_smem = RATES["smem_per_sm"][0] // smem_bytes if smem_bytes else 99
    ctas = max(0, min(ctas_reg, ctas_smem))
    return ctas * nw, ctas


class Hyperform:
    """Точка пространства поиска. params непрозрачны для конвейера; смысл им придаёт скелет."""

    __slots__ = (
        "BM",
        "BN",
        "BK",
        "WM",
        "WN",
        "STAGES",
        "GSTAGE",
        "FPREF",
        "GROUP",
        "EPI",
        "SWZ",
        "PRED",
        "MINB",
    )

    def __init__(
        self, BM, BN, BK, WM, WN, STAGES, GSTAGE, FPREF, GROUP, EPI, SWZ, PRED, MINB
    ):
        self.BM, self.BN, self.BK = BM, BN, BK
        self.WM, self.WN = WM, WN
        self.STAGES, self.GSTAGE, self.FPREF = STAGES, GSTAGE, FPREF
        self.GROUP, self.EPI, self.SWZ, self.PRED, self.MINB = (
            GROUP,
            EPI,
            SWZ,
            PRED,
            MINB,
        )

    # --- производные величины: ВЫВОДЯТСЯ из карты фрагмента, не хранятся ------------------
    @property
    def MB(self):
        return self.BM // (FRAG["tile_m"] * self.WM)

    @property
    def NB(self):
        return self.BN // (FRAG["tile_n"] * self.WN)

    @property
    def NW(self):
        return self.WM * self.WN

    @property
    def threads(self):
        return self.NW * 32

    @property
    def smem(self):
        main = self.STAGES * (self.BM + self.BN) * self.BK * 2
        epi = (16 * self.WM) * (self.BN + 8) * 2 if self.EPI else 0
        return max(main, epi)

    @property
    def kper_a(self):
        return self.BM * self.BK // 8 // self.threads

    @property
    def kper_b(self):
        return self.BN * self.BK // 8 // self.threads

    def regs_estimate(self):
        """ТОЧКА оценки: счёт по СТРУКТУРЕ ИСХОДНИКА скелета (накопители + фрагменты + подача).

        Это оценка ДАВЛЕНИЯ, а не предсказание РАСПРЕДЕЛЕНИЯ. Распределяет ptxas, и он не
        обязан совпасть: замерено, что при оценке 280 он выдаёт 254 БЕЗ ЕДИНОГО РАЗЛИВА
        (кадр 0) -- то есть перевыражает значения, лишь бы влезть в потолок ISA. Поэтому
        «стена по регистрам», объявленная ПО ЭТОЙ ОЦЕНКЕ, -- не стена, а промах оценки.
        """
        acc = self.MB * self.NB * FRAG["acc_regs_per_tile"]
        frag = (self.FPREF + 1) * (self.MB + self.NB) * 4
        stage = self.GSTAGE * (self.kper_a + self.kper_b) * 4
        return acc + frag + stage + 24

    def regs_for_verdict(self):
        """ОПТИМИСТИЧНЫЙ КРАЙ полосы -- число, которым РАЗРЕШЕНО ВЫБРАСЫВАТЬ вариант.

        ПРАВИЛО, ОПЛАЧЕННОЕ ПЯТНАДЦАТЬЮ ПОТЕРЯННЫМИ ПОБЕДИТЕЛЯМИ:
            ресурсный вердикт -- это ВЫБРАСЫВАНИЕ БЕЗ СБОРКИ. Выбрасывать по ОЦЕНКЕ можно
            только тем краем её полосы, который вариант СОХРАНЯЕТ; иначе ошибка оценки
            становится ТИХОЙ ПОТЕРЕЙ ПОБЕДИТЕЛЯ, а не громким промахом.

        Замерено правилом приёмки (tests/test_prune_acceptance.py): по ТОЧКЕ оценки
        отсекатель выбрасывал 15 из 35 замеренных победителей -- включая все три плитки,
        которые собрались (кадр 0) и обогнали сильную библиотеку. По этому краю -- ни одного.

        Полоса берётся ОДНОСТОРОННЕЙ намеренно: занижение оценки вариант не теряет, оно
        лишь оставляет лишнее, а лишнее ловится сборкой. Цена односторонности -- больше
        сборок, и это верный размен.
        """
        over = RATES["reg_estimate_over_max"][0]
        return int(self.regs_estimate() / (1.0 + over))

    # --- ЗАКОН: во сколько раз канал разделяемой длиннее тензорного ------------------------
    def conflict_factor(self):
        """Конфликтность чтения фрагментов -- функция РАСКЛАДКИ, не железа."""
        if self.SWZ:
            return 1.0
        return 1.0 if self.BK >= 64 else CONF_LEGACY[0]

    def nu_mio(self):
        MB, NB = self.MB, self.NB
        read = self.conflict_factor() * (MB + NB) / (2.0 * MB * NB)
        write = 4.0 / self.BM + 4.0 / self.BN
        return read + write

    def legal(self):
        f = FRAG
        if self.BM % (f["tile_m"] * self.WM) or self.BN % (f["tile_n"] * self.WN):
            return "не делится на плитку варпа"
        if self.MB < 1 or self.NB < 1:
            return "пустая плитка варпа"
        if self.BK % 8 or (self.BK // 8) & (self.BK // 8 - 1):
            return "BK/8 не степень двойки"
        if self.threads > 1024:
            return "нитей > 1024"
        if (self.BM * self.BK // 8) % self.threads or (
            self.BN * self.BK // 8
        ) % self.threads:
            return "плитка не делится на нити"
        if self.kper_a < 1 or self.kper_b < 1:
            return "нитей больше, чем 16-байтовых порций"
        if self.FPREF >= max(1, self.BK // 8) + 1:
            return "предвыборка глубже плитки"
        if self.smem > RATES["smem_per_sm"][0]:
            return "СТЕНА-СМЕМ"
        if self.regs_for_verdict() > 300:
            return "НЕТ-БЮДЖЕТА (оценка регистров, оптимистичный край полосы)"
        return None

    def tag(self):
        """ИМЯ ГИПЕРФОРМЫ.  Обязано быть ИНЪЕКТИВНЫМ: по контракту это ещё и `key`, то есть
        имя ядра и каталога поставки.  MINB был пропущен, и 880 гиперформ давали 440 ключей --
        две РАЗНЫЕ инстанциации (разные `__launch_bounds__`, разный вердикт ptxas по
        регистрам) писались в один каталог и различались только тем, кто последний.
        """
        return "t%dx%dk%d_w%dx%d_s%dg%df%d_G%d_E%d_Z%d%s_B%d" % (
            self.BM,
            self.BN,
            self.BK,
            self.WM,
            self.WN,
            self.STAGES,
            self.GSTAGE,
            self.FPREF,
            self.GROUP,
            self.EPI,
            self.SWZ,
            "P" if self.PRED else "",
            self.MINB,
        )

    def cfg_line(self):
        return 'CFG("%s", %d,%d,%d, %d,%d, %d,%d,%d, %d,%d,%d, %s, %d),' % (
            self.tag(),
            self.BM,
            self.BN,
            self.BK,
            self.WM,
            self.WN,
            self.STAGES,
            self.GSTAGE,
            self.FPREF,
            self.GROUP,
            self.EPI,
            self.SWZ,
            "true" if self.PRED else "false",
            self.MINB,
        )


def wave_efficiency(h, M, N):
    """Волновая квантизация: полезная доля машины на последней волне."""
    nm = -(-M // h.BM)
    nn = N // h.BN
    ctas = nm * nn
    warps, per_sm = occupancy(min(255, h.regs_for_verdict()), h.smem, h.threads)
    if per_sm == 0:
        return 0.0
    wave = RATES["sms"][0] * per_sm
    return ctas / (wave * -(-ctas // wave))


def bound_tflops(h, M, N, K, peak_tflops=125.3):
    """Граница СВЕРХУ по времени: max(тензор, MIO) + квантизация волны. Не предсказание."""
    nu = h.nu_mio()
    ideal = peak_tflops / max(1.0, nu)
    gran = (h.BM * -(-M // h.BM)) if h.PRED else M
    return ideal * wave_efficiency(h, M, N) * (M / gran)

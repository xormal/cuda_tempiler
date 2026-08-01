#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ХАРНЕСС ВРЕМЕНИ, КОТОРЫЙ ОТКАЗЫВАЕТСЯ ВРАТЬ  (tools/timeit.py)

Дисциплина замера в этом проекте выстрадана и каждый раз применяется руками, а значит через раз
забывается.  Здесь она оформлена так, что ЗАБЫТЬ НЕЛЬЗЯ: не выполненное условие не понижает
точность -- оно ОТМЕНЯЕТ результат (`valid=False`) или вовсе не даёт запустить секундомер.

ЧТО ВСТРОЕНО (и почему именно так)

1. ПАРНЫЕ ОТНОШЕНИЯ.  Варианты чередуются ВНУТРИ раунда, отношение берётся ПОРАУНДОВО, итог --
   МЕДИАНА ОТНОШЕНИЙ, а не отношение медиан.  Дрейф частоты внутри раунда достаётся обоим
   вариантам поровну и в отношении сокращается.  Доверительный интервал -- бутстрап по раундам.
   Печатаются ОБЕ величины: медиана отношений и отношение медиан.  Если они разошлись сильнее
   ширины интервала -- значит дрейф был, и это сообщается, а не заминается.

2. ДЕТЕКТОР СОСЕДА.  Чужие процессы на КАРТЕ считаются до, после и вокруг каждого раунда.
   Признак загрузки берётся НЕ из `utilization.gpu` (она двоичная и врёт), а из МОЩНОСТИ ПО
   СЕКУНДАМ: мощность опрашивается серией и сравнивается с измеренным порогом покоя карты.
   Замер с соседом -- НЕДЕЙСТВИТЕЛЕН.  Не "с оговоркой", а недействителен: `valid=False`.

3. ЧАСТОТЫ.  `nvidia-smi -lgc` перед и ГАРАНТИРОВАННОЕ `-rgc` после: try/finally + atexit +
   обработчики SIGINT/SIGTERM/SIGHUP.  Частоты печатаются ДО и ПОСЛЕ КАЖДОГО раунда: на этой
   машине карта умеет просесть до 307 МГц против 1530 и дать 60 % разброса.  Разброс частоты по
   раундам > `clk_drift_fatal` -- результат недействителен.

4. ПРОГРЕВ И ПРОВЕРКА, ЧТО ОН СОСТОЯЛСЯ.  Мало прогреть -- надо доказать: первый раунд не должен
   отличаться от остальных сильнее, чем остальные отличаются между собой (робастный признак по
   медиане и MAD).  Не доказан -- результат недействителен.

5. ГЕЙТ КОРРЕКТНОСТИ ПЕРЕД ВРЕМЕНЕМ.  `check` обязателен и вызывается ДО секундомера.  Не прошёл
   -- секундомер не запускается вовсе.  `check=None` -- отказ запуска.  Явный `Harness.NO_CHECK`
   разрешён, но попадает в раздел "НЕ РАЗОБРАНО" и в заголовок отчёта.

6. РАЗДЕЛ "НЕ РАЗОБРАНО".  Всё, что инструмент не смог установить (поле nvidia-smi не
   разобралось, порог покоя не откалиброван, точка входа не подтверждена, частоты не
   зафиксированы), печатается СПИСКОМ.  Пустой список замечаний при непустом списке
   неразобранного НЕ означает "чисто" -- так и печатается.

ЗАПУСК
    python3 tools/timeit.py --selftest              # самопроверка, БЕЗ железа (чистый CPU)
    python3 tools/timeit.py --precheck --card 1     # только гейты среды: можно ли сейчас мерить
    # как библиотека:
    from timeit_harness import Harness              # см. tools/anchor_fwd_sdpa.py
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------------------------
# ЭТОТ ФАЙЛ НАЗЫВАЕТСЯ timeit.py И ТЕМ САМЫМ ЗАСЛОНЯЕТ СТАНДАРТНЫЙ МОДУЛЬ, КОТОРЫЙ ИМПОРТИРУЕТ
# torch (`from timeit import default_timer`).  Когда файл запускают как скрипт, его каталог
# становится sys.path[0], и ленивый `import torch` в _cuda_timer() падает:
#   ImportError: cannot import name 'default_timer' from 'timeit' (.../tempo/tools/timeit.py)
# ВОСПРОИЗВЕДЕНО 2026-08-01.  То есть БОЕВОЙ режим (единственный, где вообще есть секундомер)
# не запускался НИКОГДА -- отказ был бы принят за "карта занята".  Каталог скрипта уходит из
# sys.path первым действием; тот же приём уже стоит в ncu.py, smem_lint_verify.py,
# phase_plumbing_check.py, phase_selfcheck_fwd_ws.py.
_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE_DIR]

# ---------------------------------------------------------------------------------------------
# РАЗБОР ЧИСЕЛ: русская локаль ломала CSV nvidia-smi/ncu -- разделитель тысяч НЕРАЗРЫВНЫЙ пробел.
# Наивный разбор молча выбрасывал всё >= 1000, то есть ровно интересные значения.  Здесь:
#   * подпроцессы запускаются с LC_ALL=C/LANG=C (лечение причины);
#   * парсер всё равно терпит NBSP и запятую (лечение следствия);
#   * ЧТО НЕ РАЗОБРАЛОСЬ -- ПОПАДАЕТ В СПИСОК, а не превращается в 0.
# ---------------------------------------------------------------------------------------------
_SPACES = "       \t"
_NA = {
    "",
    "n/a",
    "[n/a]",
    "notsupported",
    "[notsupported]",
    "unknown",
    "-",
    "[unknown]",
}
_THOUS = re.compile(r"^\d{1,3}(,\d{3})+$")


def parse_num(raw, unparsed=None, what=""):
    """Число из поля nvidia-smi.  Не разобралось -> None И запись в `unparsed` (никогда не 0)."""
    if raw is None:
        if unparsed is not None:
            unparsed.append(f"{what}: поле отсутствует")
        return None
    t = str(raw)
    for ch in _SPACES:
        t = t.replace(ch, "")
    if t.lower() in _NA:
        if unparsed is not None:
            unparsed.append(f"{what}: значение '{str(raw).strip()}' (не число)")
        return None
    if _THOUS.match(t):  # 1,530 -- английский разделитель тысяч
        t = t.replace(",", "")
    elif t.count(",") == 1 and "." not in t:  # 51,11 -- русская дробная запятая
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        if unparsed is not None:
            unparsed.append(
                f"{what}: строка '{str(raw).strip()}' не разобрана как число"
            )
        return None


def _env_c():
    e = dict(os.environ)
    e["LC_ALL"] = "C"
    e["LANG"] = "C"
    return e


# ---------------------------------------------------------------------------------------------
# ОПРОС КАРТЫ.  Всё через одну точку -- её можно подменить в самопроверке (`runner`).
# ---------------------------------------------------------------------------------------------
class Smi:
    def __init__(self, runner=None, timeout=15):
        self.runner = runner or self._real
        self.timeout = timeout
        self.unparsed = []

    def _real(self, args):
        try:
            p = subprocess.run(
                ["nvidia-smi"] + args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_env_c(),
            )
            return p.returncode, p.stdout, p.stderr
        except FileNotFoundError:
            return 127, "", "nvidia-smi не найден"
        except subprocess.TimeoutExpired:
            return 124, "", "nvidia-smi не ответил"

    def _q(self, fields, card=None, extra=None):
        args = [f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        if card is not None:
            args += ["-i", str(card)]
        if extra:
            args += extra
        rc, out, err = self.runner(args)
        if rc != 0:
            self.unparsed.append(f"nvidia-smi {fields}: rc={rc} {err.strip()[:120]}")
            return []
        return [ln for ln in out.splitlines() if ln.strip()]

    def state(self, card):
        """Мгновенное состояние карты.  Поля, которые не разобрались, приходят как None."""
        rows = self._q(
            "clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,pstate",
            card,
        )
        if not rows:
            return {
                "clk": None,
                "clk_max": None,
                "power": None,
                "temp": None,
                "mem": None,
                "pstate": None,
            }
        f = [x.strip() for x in rows[0].split(",")]
        f += [None] * (6 - len(f))
        return {
            "clk": parse_num(f[0], self.unparsed, f"clocks.sm[{card}]"),
            "clk_max": parse_num(f[1], self.unparsed, f"clocks.max.sm[{card}]"),
            "power": parse_num(f[2], self.unparsed, f"power.draw[{card}]"),
            "temp": parse_num(f[3], self.unparsed, f"temperature.gpu[{card}]"),
            "mem": parse_num(f[4], self.unparsed, f"memory.used[{card}]"),
            "pstate": f[5],
        }

    def uuid(self, card):
        rows = self._q("uuid", card)
        return rows[0].strip() if rows else None

    def foreign(self, card, my_pid=None):
        """Чужие вычислительные процессы НА ЭТОЙ карте: [(pid, МиБ)].  Свой pid исключён."""
        uu = self.uuid(card)
        rc, out, err = self.runner(
            [
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        if rc != 0:
            self.unparsed.append(f"compute-apps: rc={rc} {err.strip()[:120]}")
            return None  # None = НЕ УСТАНОВЛЕНО (не "ноль")
        if uu is None:
            self.unparsed.append(f"uuid карты {card} не получен -- сосед не проверен")
            return None
        me = str(my_pid if my_pid is not None else os.getpid())
        res = []
        for ln in out.splitlines():
            if not ln.strip():
                continue
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) < 2:
                self.unparsed.append(
                    f"compute-apps: строка '{ln.strip()[:60]}' не разобрана"
                )
                continue
            if parts[0] != uu or parts[1] == me:
                continue
            res.append(
                (
                    parts[1],
                    parse_num(
                        parts[2] if len(parts) > 2 else None,
                        self.unparsed,
                        "used_memory",
                    ),
                )
            )
        return res

    def power_series(self, card, seconds=3.0, hz=10.0):
        """МОЩНОСТЬ ПО СЕКУНДАМ -- честный признак загрузки (utilization.gpu двоична и врёт).

        `-c N` этот драйвер с `--query-gpu` НЕ принимает (проверено: 'Option --query-gpu is not
        recognized'), поэтому поток `-lms` читается и обрывается нами.  Не пошло -- запасной
        путь одиночными опросами, и КАКОЙ путь сработал, возвращается вторым значением."""
        n = max(2, int(round(seconds * hz)))
        vals = []
        p = None
        try:
            p = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    str(card),
                    "--format=csv,noheader,nounits",
                    "--query-gpu=power.draw",
                    "-lms",
                    str(max(50, int(1000.0 / hz))),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_env_c(),
            )
            t0 = time.time()
            while len(vals) < n and time.time() - t0 < seconds * 3 + 10:
                ln = p.stdout.readline()
                if not ln:
                    break
                v = parse_num(ln, self.unparsed, "power.draw(серия)")
                if v is not None:
                    vals.append(v)
        except Exception as e:  # noqa: BLE001
            self.unparsed.append(f"поток мощности: {type(e).__name__}: {e}")
        finally:
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:  # noqa: BLE001
                        pass
        if len(vals) >= max(2, n // 2):
            return vals, "-lms"
        vals, t0 = [], time.time()  # запасной путь: опрос в цикле
        while len(vals) < n and time.time() - t0 < seconds * 3 + 5:
            v = self.state(card)["power"]
            if v is not None:
                vals.append(v)
            time.sleep(1.0 / hz)
        return vals, "цикл"


# ---------------------------------------------------------------------------------------------
# ФИКСАЦИЯ ЧАСТОТ.  Снятие гарантировано: try/finally + atexit + сигналы.
# ---------------------------------------------------------------------------------------------
_ACTIVE_LOCKS = []
_SIG_INSTALLED = False


def _release_all(*_a):
    for lk in list(_ACTIVE_LOCKS):
        try:
            lk.release()
        except Exception:
            pass


def _install_signal_handlers():
    global _SIG_INSTALLED
    if _SIG_INSTALLED:
        return
    _SIG_INSTALLED = True
    atexit.register(_release_all)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            prev = signal.getsignal(s)

            def h(sig, frm, _prev=prev):
                _release_all()
                if callable(_prev):
                    return _prev(sig, frm)
                if _prev == signal.SIG_DFL:
                    signal.signal(sig, signal.SIG_DFL)
                    os.kill(os.getpid(), sig)

            signal.signal(s, h)
        except (ValueError, OSError):
            pass  # не главный поток -- переживём


class ClockLock:
    """`nvidia-smi -lgc` с ГАРАНТИРОВАННЫМ `-rgc`.  Пароль читается из переменной среды и
    НИКОГДА не печатается и не попадает в argv (уходит только в stdin sudo)."""

    def __init__(self, card, mhz, pass_env="FA2_SUDO_PASS", runner=None):
        self.card, self.mhz, self.pass_env = card, mhz, pass_env
        self.runner = runner or self._real
        self.locked = False
        self.released = False
        self.release_calls = 0
        self.reason = None

    def _real(self, args):
        pw = os.environ.get(self.pass_env)
        try:
            if pw:
                p = subprocess.run(
                    ["sudo", "-S", "-p", ""] + args,
                    input=pw + "\n",
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=_env_c(),
                )
            else:
                p = subprocess.run(
                    args, capture_output=True, text=True, timeout=60, env=_env_c()
                )
            return p.returncode, p.stdout, p.stderr
        except Exception as e:  # noqa: BLE001
            return 1, "", str(e)[:120]

    def acquire(self):
        _install_signal_handlers()
        if not self.mhz:
            self.reason = "фиксация не запрошена"
            return False
        rc, out, err = self.runner(
            ["nvidia-smi", "-i", str(self.card), "-lgc", f"{self.mhz},{self.mhz}"]
        )
        if rc == 0:
            self.locked = True
            _ACTIVE_LOCKS.append(self)
        else:
            self.reason = f"rc={rc} {(err or out).strip()[:100]}"
        return self.locked

    def release(self):
        self.release_calls += 1
        if not self.locked or self.released:
            return
        rc, out, err = self.runner(["nvidia-smi", "-i", str(self.card), "-rgc"])
        self.released = rc == 0
        if not self.released:
            self.reason = f"СНЯТИЕ НЕ УДАЛОСЬ: rc={rc} {(err or out).strip()[:100]}"
            print(
                f"!!! ЧАСТОТЫ КАРТЫ {self.card} ОСТАЛИСЬ ЗАФИКСИРОВАНЫ: {self.reason}\n"
                f"!!! снять вручную: sudo nvidia-smi -i {self.card} -rgc",
                file=sys.stderr,
            )
        if self in _ACTIVE_LOCKS:
            _ACTIVE_LOCKS.remove(self)


# ---------------------------------------------------------------------------------------------
# СТАТИСТИКА
# ---------------------------------------------------------------------------------------------
def median(xs):
    return statistics.median(xs)


def mad(xs):
    m = median(xs)
    return median([abs(x - m) for x in xs])


def bootstrap_ci(vals, stat=median, iters=10000, alpha=0.05, seed=12345):
    """Процентильный бутстрап по РАУНДАМ.  Возвращает (lo, hi, se)."""
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    rnd = random.Random(seed)
    out = []
    for _ in range(iters):
        out.append(stat([vals[rnd.randrange(n)] for _ in range(n)]))
    out.sort()
    lo = out[max(0, int(math.floor(alpha / 2 * iters)) - 1)]
    hi = out[min(iters - 1, int(math.ceil((1 - alpha / 2) * iters)) - 1)]
    return (lo, hi, statistics.pstdev(out))


def warmup_ok(series, k=3.0, rel=0.03):
    """СОСТОЯЛСЯ ЛИ ПРОГРЕВ: первый раунд не должен отличаться от остальных сильнее, чем
    остальные между собой.  Возвращает (ok, отклонение_первого, разброс_остальных)."""
    if len(series) < 4:
        return (None, float("nan"), float("nan"))
    rest = series[1:]
    m = median(rest)
    spread = mad(rest)
    d = abs(series[0] - m)
    if m == 0:
        return (None, d, spread)
    ok = d <= max(k * spread, rel * abs(m))
    return (bool(ok), d / abs(m), spread / abs(m))


# ---------------------------------------------------------------------------------------------
# ПРИГОВОР
# ---------------------------------------------------------------------------------------------
class Verdict:
    def __init__(self):
        self.valid = True
        self.fatal = []  # почему НЕДЕЙСТВИТЕЛЕН
        self.warn = []  # что подозрительно
        self.unparsed = []  # ЧТО НЕ УСТАНОВЛЕНО (главный раздел: неполнота != чистота)

    def kill(self, why):
        self.valid = False
        self.fatal.append(why)

    def to_dict(self):
        return {
            "valid": self.valid,
            "fatal": self.fatal,
            "warn": self.warn,
            "unparsed": self.unparsed,
        }


class Result:
    def __init__(self):
        self.label = ""
        self.base = ""
        self.variants = []
        self.raw = {}  # имя -> [мс по раундам]
        self.ratios = {}  # имя -> [пораундовые отношения base/вариант]
        self.summary = {}  # имя -> {...}
        self.rounds_state = []
        self.verdict = Verdict()
        self.checked = None
        self.check_msg = ""
        self.env = {}
        self.timed_calls = 0

    def to_dict(self):
        return {
            "label": self.label,
            "base": self.base,
            "raw_ms": self.raw,
            "ratios": self.ratios,
            "summary": self.summary,
            "rounds_state": self.rounds_state,
            "verdict": self.verdict.to_dict(),
            "checked": self.checked,
            "check_msg": self.check_msg,
            "env": self.env,
        }

    # ---- отчёт -------------------------------------------------------------------------------
    def report(self):
        L = []
        head = "ДЕЙСТВИТЕЛЕН" if self.verdict.valid else "*** НЕДЕЙСТВИТЕЛЕН ***"
        if self.checked is False:
            head += "  [СВЕРКА НЕ ПРОВОДИЛАСЬ]"
        L.append(f"=== {self.label or 'замер'} -- {head} ===")
        if self.env:
            L.append("среда: " + ", ".join(f"{k}={v}" for k, v in self.env.items()))
        if self.checked is not None:
            L.append(
                f"сверка ДО секундомера: {'ПРОШЛА' if self.checked else 'НЕ ПРОШЛА'}"
                + (f" -- {self.check_msg}" if self.check_msg else "")
            )
        if self.summary:
            L.append(
                f"{'вариант':<18}{'мс(мед)':>10}{'мин':>9}{'разбр%':>8}"
                f"{'отн(мед)':>10}{'ДИ 95%':>18}{'отн.медиан':>11}{'прогрев':>9}"
            )
            for v in self.variants:
                s = self.summary[v]
                ci = (
                    f"[{s['ci_lo']:.4f},{s['ci_hi']:.4f}]"
                    if s["ci_lo"] == s["ci_lo"]
                    else "--"
                )
                wu = (
                    ("ok" if s["warmup_ok"] else "НЕТ")
                    if s["warmup_ok"] is not None
                    else "?"
                )
                L.append(
                    f"{v:<18}{s['median_ms']:>10.4f}{s['min_ms']:>9.4f}"
                    f"{s['spread_pct']:>8.2f}{s['ratio_median']:>10.4f}{ci:>18}"
                    f"{s['ratio_of_medians']:>11.4f}{wu:>9}"
                )
            L.append(
                f"отношение = t[{self.base}] / t[вариант]   (>1 = вариант БЫСТРЕЕ базы)"
            )
        for w in self.verdict.fatal:
            L.append(f"ОТМЕНА: {w}")
        for w in self.verdict.warn:
            L.append(f"замечание: {w}")
        L.append(
            "--- НЕ РАЗОБРАНО / НЕ УСТАНОВЛЕНО (%d) ---" % len(self.verdict.unparsed)
        )
        if self.verdict.unparsed:
            for u in self.verdict.unparsed:
                L.append(f"  ? {u}")
            if not self.verdict.fatal and not self.verdict.warn:
                L.append(
                    "  ВНИМАНИЕ: замечаний нет, но список выше НЕ ПУСТ -- это НЕ 'чисто'."
                )
        else:
            L.append("  (пусто)")
        return "\n".join(L)


# ---------------------------------------------------------------------------------------------
# ХАРНЕСС
# ---------------------------------------------------------------------------------------------
class NoCheck:
    pass


class Harness:
    NO_CHECK = NoCheck

    def __init__(
        self,
        card=1,
        rounds=17,
        warmup=8,
        iters=1,
        lock_mhz=1530,
        idle_watts=None,
        idle_probe_s=3.0,
        clk_drift_warn=0.01,
        clk_drift_fatal=0.03,
        smi=None,
        timer=None,
        force=False,
        sudo_pass_env="FA2_SUDO_PASS",
        idle_calib=None,
        allow_no_lock=False,
    ):
        self.card = card
        self.rounds = rounds
        self.warmup = warmup
        self.iters = iters
        self.lock_mhz = lock_mhz
        self.idle_probe_s = idle_probe_s
        self.clk_drift_warn = clk_drift_warn
        self.clk_drift_fatal = clk_drift_fatal
        self.smi = smi or Smi()
        self.timer = timer  # None -> события CUDA; иначе callable(fn,iters)->мс
        self.force = force
        self.sudo_pass_env = sudo_pass_env
        self.allow_no_lock = allow_no_lock
        self.idle_calib = idle_calib or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            f"idle_power_card{card}.json",
        )
        # Порог покоя: измеренный (калибровка) либо ПО УМОЛЧАНИЮ.  Умолчание -- V100-SXM2 в покое
        # даёт 40-55 Вт (замерено на всех четырёх картах этой машины), 70 Вт -- уже чужая работа.
        # Ставится не для красоты: без порога гейт мощности МОЛЧА выключается, а это ровно тот
        # отказ, который выглядит как "чисто".
        self.default_idle_watts = 70.0
        self.idle_calibrated = True
        self.idle_watts = idle_watts if idle_watts is not None else self._load_idle()
        if self.idle_watts is None:
            self.idle_watts = self.default_idle_watts
            self.idle_calibrated = False

    # -- порог покоя ---------------------------------------------------------------------------
    def _load_idle(self):
        try:
            with open(self.idle_calib) as f:
                d = json.load(f)
            return float(d["idle_watts_p95"])
        except Exception:  # noqa: BLE001
            return None

    def calibrate_idle(self, seconds=20.0, hz=5.0, force=False):
        """Порог покоя КАРТЫ -- измеряется, а не назначается.  Запускать на ПУСТОЙ карте:
        калибровка при соседе впечатала бы ЧУЖУЮ нагрузку в порог и навсегда ослепила гейт."""
        fo = self.smi.foreign(self.card)
        if fo and not force:
            print(
                f"КАЛИБРОВКА ОТМЕНЕНА: на карте {self.card} чужих процессов {len(fo)} -- "
                "порог покоя, снятый при соседе, выключил бы детектор соседа",
                file=sys.stderr,
            )
            return None
        if fo is None and not force:
            print(
                f"КАЛИБРОВКА ОТМЕНЕНА: состав процессов карты {self.card} не установлен",
                file=sys.stderr,
            )
            return None
        vals, how = self.smi.power_series(self.card, seconds, hz)
        if not vals:
            return None
        vals_sorted = sorted(vals)
        p95 = vals_sorted[min(len(vals) - 1, int(0.95 * len(vals)))]
        d = {
            "card": self.card,
            "n": len(vals),
            "how": how,
            "median": median(vals),
            "p95": p95,
            "idle_watts_p95": p95,
            "foreign_at_calib": fo,
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.idle_calib)), exist_ok=True)
        with open(self.idle_calib, "w") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
        self.idle_watts = p95
        return d

    # -- гейт среды ----------------------------------------------------------------------------
    def precheck(self, verdict=None):
        """ГЕЙТ СРЕДЫ.  Возвращает (можно_мерить, verdict, факты).  Ничего на карте не занимает."""
        v = verdict or Verdict()
        facts = {}
        fo = self.smi.foreign(self.card)
        facts["foreign_before"] = fo
        if fo is None:
            v.kill(
                f"состав процессов карты {self.card} НЕ УСТАНОВЛЕН (nvidia-smi не ответил)"
            )
            v.unparsed.append("сосед: список процессов не получен")
        elif fo:
            v.kill(
                f"на карте {self.card} ЧУЖИХ ПРОЦЕССОВ: {len(fo)} "
                f"({', '.join('pid ' + p + (' ' + str(m) + 'МиБ' if m else '') for p, m in fo)})"
                " -- замер времени недействителен"
            )
        pw, how = self.smi.power_series(self.card, self.idle_probe_s)
        facts["idle_power_n"] = len(pw)
        facts["idle_power_how"] = how
        if not pw:
            v.kill(
                f"мощность карты {self.card} НЕ ИЗМЕРЕНА -- загрузку соседом установить нечем"
            )
            v.unparsed.append("мощность: серия пуста")
        else:
            facts["idle_power_med"] = median(pw)
            facts["idle_power_max"] = max(pw)
            if not self.idle_calibrated:
                v.unparsed.append(
                    f"порог покоя карты {self.card} НЕ ОТКАЛИБРОВАН ({self.idle_calib} нет): "
                    f"взят умолчательный {self.default_idle_watts:.0f} Вт, сейчас "
                    f"{median(pw):.1f} Вт (калибровка: --calibrate-idle на пустой карте)"
                )
            if median(pw) > self.idle_watts:
                v.kill(
                    f"карта {self.card} НЕ В ПОКОЕ: {median(pw):.1f} Вт по серии из {len(pw)} "
                    f"против порога покоя {self.idle_watts:.1f} Вт "
                    "(мощность по секундам, а не utilization.gpu)"
                )
        st = self.smi.state(self.card)
        facts["state_before"] = st
        for k in ("clk", "power"):
            if st.get(k) is None:
                v.unparsed.append(f"поле {k} карты {self.card} не разобрано")
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            v.unparsed.append(
                f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: соответствие "
                f"логического устройства torch физической карте {self.card} НЕ ПРОВЕРЕНО "
                "инструментом -- задавайте card= физическим индексом"
            )
        v.unparsed.extend(self.smi.unparsed)
        self.smi.unparsed = []
        return (v.valid, v, facts)

    # -- секундомер ----------------------------------------------------------------------------
    def _cuda_timer(self):
        import torch

        def t(fn, iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        return t

    def compare(self, variants, base, check, label="", entry_probe=None):
        """Парный замер.  variants: {имя: callable}.  base -- имя базы.  check -- ОБЯЗАТЕЛЕН.

        Порядок ЖЁСТКИЙ: гейт среды -> сверка -> фиксация частот -> прогрев -> раунды.
        Секундомер не запускается, пока не пройдены первые два.
        """
        r = Result()
        r.label = label
        r.base = base
        r.variants = list(variants.keys())
        v = r.verdict
        r.env = {
            "card": self.card,
            "rounds": self.rounds,
            "warmup": self.warmup,
            "iters": self.iters,
            "pid": os.getpid(),
            "lock_mhz": self.lock_mhz or "нет",
        }
        if base not in variants:
            v.kill(f"база '{base}' отсутствует среди вариантов")
            return r

        # 1. ГЕЙТ СРЕДЫ ------------------------------------------------------------------------
        ok_env, v, facts = self.precheck(v)
        r.env["мощность_покоя_Вт"] = round(facts.get("idle_power_med", float("nan")), 1)
        r.env["чужих_процессов"] = (
            len(facts["foreign_before"])
            if facts.get("foreign_before") is not None
            else "?"
        )
        if not ok_env and not self.force:
            v.unparsed.append("СЕКУНДОМЕР НЕ ЗАПУСКАЛСЯ: гейт среды не пройден")
            return r
        if not ok_env:
            v.warn.append(
                "--force: замер выполнен ВОПРЕКИ гейту среды и остаётся НЕДЕЙСТВИТЕЛЬНЫМ"
            )

        # 2. СВЕРКА ДО ВРЕМЕНИ -----------------------------------------------------------------
        if check is None:
            v.kill(
                "функция сверки не задана: замер без гейта корректности запрещён "
                "(явный отказ -- Harness.NO_CHECK)"
            )
            v.unparsed.append("СЕКУНДОМЕР НЕ ЗАПУСКАЛСЯ: нет функции сверки")
            return r
        if check is Harness.NO_CHECK:
            r.checked = False
            v.unparsed.append(
                "СВЕРКА НЕ ПРОВОДИЛАСЬ (запрошен Harness.NO_CHECK): числа ниже "
                "не подтверждают, что варианты считают одно и то же"
            )
        else:
            try:
                res = check()
                if isinstance(res, tuple):
                    ok, msg = res[0], str(res[1])
                else:
                    ok, msg = bool(res), ""
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"исключение: {type(e).__name__}: {e}"
            r.checked, r.check_msg = bool(ok), msg
            if not ok:
                v.kill(f"ГЕЙТ КОРРЕКТНОСТИ НЕ ПРОЙДЕН: {msg}")
                v.unparsed.append("СЕКУНДОМЕР НЕ ЗАПУСКАЛСЯ: сверка не прошла")
                return r

        # 2б. ТОЧКА ВХОДА ----------------------------------------------------------------------
        if entry_probe:
            seen = {}
            for name, probe in entry_probe.items():
                try:
                    seen[name] = probe()
                except Exception as e:  # noqa: BLE001
                    seen[name] = f"ошибка: {e}"
            r.env["точка_входа"] = seen
            known = {k: str(x) for k, x in seen.items() if x is not None}
            for k in seen:
                if seen[k] is None:
                    v.unparsed.append(
                        f"точка входа варианта '{k}' НЕ УСТАНОВЛЕНА "
                        "(проба вернула None) -- совпадение с базой не исключено"
                    )
            if len(set(known.values())) < len(known):
                v.kill(
                    f"ТОЧКА ВХОДА СОВПАЛА У РАЗНЫХ ВАРИАНТОВ: {known} -- A/B не состоялся"
                )
                return r
        else:
            v.unparsed.append(
                "точка входа НЕ ПОДТВЕРЖДЕНА: entry_probe не задан "
                "(вариант мог не отличаться от базы ничем, кроме имени)"
            )

        # 3. ЧАСТОТЫ ---------------------------------------------------------------------------
        lock = ClockLock(self.card, self.lock_mhz, self.sudo_pass_env)
        timer = self.timer or self._cuda_timer()
        try:
            got = lock.acquire()
            r.env["частоты_зафиксированы"] = bool(got)
            if not got and not self.lock_mhz:
                # фиксация НЕ ЗАПРАШИВАЛАСЬ -- это выбор вызывающего, но он обязан быть виден
                v.unparsed.append(
                    f"частоты карты {self.card} НЕ ФИКСИРОВАЛИСЬ (lock_mhz не задан): дрейф "
                    "ловится только измерением частоты по раундам, до замера ничего не гарантировано"
                )
            elif not got:
                msg = (
                    f"частоты карты {self.card} НЕ ЗАФИКСИРОВАНЫ ({lock.reason}); "
                    "на этой машине разброс частоты даёт до 60 %"
                )
                if self.allow_no_lock:
                    v.warn.append(msg)
                    v.unparsed.append("частоты: фиксация не подтверждена")
                else:
                    v.kill(msg)
                    v.unparsed.append(
                        "СЕКУНДОМЕР НЕ ЗАПУСКАЛСЯ: частоты не зафиксированы "
                        "(разрешить: allow_no_lock=True)"
                    )
                    return r

            # 4. ПРОГРЕВ -----------------------------------------------------------------------
            for _ in range(self.warmup):
                for fn in variants.values():
                    fn()
                    r.timed_calls += 0

            # 5. РАУНДЫ ------------------------------------------------------------------------
            raw = {k: [] for k in variants}
            for i in range(self.rounds):
                s0 = self.smi.state(self.card)
                f0 = self.smi.foreign(self.card)
                for name, fn in variants.items():  # ЧЕРЕДОВАНИЕ ВНУТРИ РАУНДА
                    ms = timer(fn, self.iters)
                    raw[name].append(ms)
                    r.timed_calls += 1
                s1 = self.smi.state(self.card)
                f1 = self.smi.foreign(self.card)
                r.rounds_state.append(
                    {
                        "round": i,
                        "clk_before": s0["clk"],
                        "clk_after": s1["clk"],
                        "pw_before": s0["power"],
                        "pw_after": s1["power"],
                        "foreign_before": None if f0 is None else len(f0),
                        "foreign_after": None if f1 is None else len(f1),
                    }
                )
        finally:
            lock.release()  # ГАРАНТИЯ
            r.env["частоты_сняты"] = bool(lock.released or not lock.locked)

        # 6. РАЗБОР ------------------------------------------------------------------------------
        r.raw = raw
        b = raw[base]
        for name in variants:
            xs = raw[name]
            rat = [b[i] / xs[i] for i in range(len(xs))]
            lo, hi, se = bootstrap_ci(rat)
            wok, wdev, wspr = warmup_ok(xs)
            r.ratios[name] = rat
            r.summary[name] = {
                "median_ms": median(xs),
                "min_ms": min(xs),
                "spread_pct": 100.0 * (max(xs) - min(xs)) / max(median(xs), 1e-12),
                "ratio_median": median(rat),
                "ci_lo": lo,
                "ci_hi": hi,
                "boot_se": se,
                "ratio_of_medians": median(b) / median(xs),
                "warmup_ok": wok,
                "warmup_dev": wdev,
                "warmup_spread": wspr,
            }
            if wok is False:
                v.kill(
                    f"ПРОГРЕВ НЕ СОСТОЯЛСЯ для '{name}': первый раунд отклонён на "
                    f"{100 * wdev:.1f} % при разбросе остальных {100 * wspr:.1f} % "
                    "(увеличьте warmup)"
                )
            elif wok is None:
                v.unparsed.append(f"прогрев для '{name}' не проверен: раундов < 4")
            # расхождение двух способов усреднения = признак дрейфа
            if name != base and se == se:
                d = abs(
                    r.summary[name]["ratio_median"]
                    - r.summary[name]["ratio_of_medians"]
                )
                if d > (hi - lo):
                    v.warn.append(
                        f"'{name}': медиана отношений {r.summary[name]['ratio_median']:.4f} и "
                        f"отношение медиан {r.summary[name]['ratio_of_medians']:.4f} разошлись "
                        f"сильнее ширины ДИ -- в раундах был дрейф (верно первое)"
                    )

        # частоты и соседи по раундам
        clks = [
            s["clk_before"] for s in r.rounds_state if s["clk_before"] is not None
        ] + [s["clk_after"] for s in r.rounds_state if s["clk_after"] is not None]
        if clks:
            drift = (max(clks) - min(clks)) / max(max(clks), 1e-9)
            r.env["частоты_МГц"] = f"{min(clks):.0f}..{max(clks):.0f}"
            if drift > self.clk_drift_fatal:
                v.kill(
                    f"ЧАСТОТА ПЛЫЛА: {min(clks):.0f}..{max(clks):.0f} МГц "
                    f"({100 * drift:.1f} % > {100 * self.clk_drift_fatal:.0f} %)"
                )
            elif drift > self.clk_drift_warn:
                v.warn.append(
                    f"частота {min(clks):.0f}..{max(clks):.0f} МГц ({100 * drift:.1f} %)"
                )
        else:
            v.unparsed.append("частоты по раундам не получены -- дрейф не проверен")
        fo_after = [s["foreign_after"] for s in r.rounds_state] + [
            s["foreign_before"] for s in r.rounds_state
        ]
        if any(x is None for x in fo_after):
            v.unparsed.append("состав процессов в части раундов не установлен")
        elif any(x > 0 for x in fo_after):
            v.kill(
                f"СОСЕД ПОЯВИЛСЯ ВО ВРЕМЯ ЗАМЕРА (max {max(fo_after)} чужих процессов)"
            )
        v.unparsed.extend(self.smi.unparsed)
        self.smi.unparsed = []
        return r


# =============================================================================================
# САМОПРОВЕРКА -- полностью на CPU, железо не трогается.
# =============================================================================================
class FakeSmi(Smi):
    """Подставная карта: задаются частоты по вызовам, мощность и список чужих процессов."""

    def __init__(
        self, clk=1530.0, power=45.0, foreign=(), clk_seq=None, broken_fields=False
    ):
        super().__init__(runner=lambda a: (0, "", ""))
        self.clk, self.power, self._foreign = clk, power, list(foreign)
        self.clk_seq, self.i = clk_seq, 0
        self.broken_fields = broken_fields

    def state(self, card):
        c = self.clk
        if self.clk_seq:
            c = self.clk_seq[min(self.i, len(self.clk_seq) - 1)]
            self.i += 1
        if self.broken_fields:
            return {
                "clk": parse_num("N/A", self.unparsed, "clocks.sm[fake]"),
                "clk_max": 1530.0,
                "power": parse_num("1 530", self.unparsed, "power[fake]"),
                "temp": 40.0,
                "mem": 0.0,
                "pstate": "P0",
            }
        return {
            "clk": c,
            "clk_max": 1530.0,
            "power": self.power,
            "temp": 40.0,
            "mem": 300.0,
            "pstate": "P0",
        }

    def uuid(self, card):
        return "GPU-FAKE"

    def foreign(self, card, my_pid=None):
        return list(self._foreign)

    def power_series(self, card, seconds=3.0, hz=10.0):
        return ([self.power] * max(2, int(seconds * hz)), "fake")


def _fake_workload(
    true_ratio=1.09,
    rounds=17,
    seed=0,
    drift_ramp=1.6,
    spike_p=0.06,
    noise=0.02,
    blocked=False,
    first_round_cold=1.0,
):
    """Синтетика с ИЗВЕСТНЫМ ответом: база = 1.0 мс, вариант = 1/true_ratio мс.
    Поверх -- общий пораундовый дрейф частоты (рампа), редкие всплески соседа и шум измерения."""
    rnd = random.Random(seed)
    base_t, var_t = 1.0, 1.0 / true_ratio
    raw = {"base": [], "var": []}
    for i in range(rounds):
        d = 1.0 / (1.0 + (drift_ramp - 1.0) * i / max(1, rounds - 1))  # разгон частоты
        for name, t in (("base", base_t), ("var", var_t)):
            f = d
            if blocked:  # НЕ чередуем: сперва все base, потом все var
                f = 1.0 / (
                    1.0
                    + (drift_ramp - 1.0)
                    * ((i if name == "base" else rounds + i) / max(1, 2 * rounds - 1))
                )
            if rnd.random() < spike_p:
                f *= 1.0 + rnd.random() * 0.8
            if i == 0:
                f *= first_round_cold
            raw[name].append(t * f * (1.0 + rnd.gauss(0, noise)))
    return raw


def _est(raw):
    b, x = raw["base"], raw["var"]
    rat = [b[i] / x[i] for i in range(len(x))]
    return median(rat), median(b) / median(x), rat


def selftest(verbose=True):
    P = []

    def chk(name, ok, note=""):
        P.append((name, bool(ok), note))
        if verbose:
            print(
                f"[{'ok ' if ok else 'ПРОВАЛ'}] {name}" + (f"   {note}" if note else "")
            )

    # --- Т1. ЯКОРЬ ОЦЕНИВАТЕЛЯ: восстанавливаем ИЗВЕСТНОЕ отношение 1.09 -----------------------
    TRUE = 1.09
    reps, cov, errs_p, errs_m, errs_blocked = 200, 0, [], [], []
    for s in range(reps):
        raw = _fake_workload(TRUE, rounds=17, seed=s)
        mr, rm, rat = _est(raw)
        lo, hi, _ = bootstrap_ci(rat, iters=1000, seed=1000 + s)
        cov += int(lo <= TRUE <= hi)
        errs_p.append(mr - TRUE)
        errs_m.append(rm - TRUE)
        rawb = _fake_workload(TRUE, rounds=17, seed=s, blocked=True)
        errs_blocked.append(_est(rawb)[0] - TRUE)
    bias = median(errs_p)
    rmse_p = math.sqrt(sum(e * e for e in errs_p) / reps)
    rmse_m = math.sqrt(sum(e * e for e in errs_m) / reps)
    rmse_b = math.sqrt(sum(e * e for e in errs_blocked) / reps)
    chk(
        "Т1 медиана парных отношений восстанавливает истинное 1.0900",
        abs(bias) < 0.005,
        f"смещение {bias:+.4f} ({100 * bias / TRUE:+.2f} %), СКО {rmse_p:.4f}",
    )
    chk(
        "Т1б покрытие бутстрап-ДИ 95 % не ниже 0.90",
        cov / reps >= 0.90,
        f"покрытие {cov / reps:.3f} на {reps} повторах",
    )
    chk(
        "Т1в чередование ВНУТРИ раунда бьёт блочный порядок (тот же дрейф)",
        rmse_b > rmse_p * 1.5,
        f"СКО блочного {rmse_b:.4f} против парного {rmse_p:.4f} "
        f"(x{rmse_b / max(rmse_p, 1e-9):.1f})",
    )
    chk(
        "Т1г медиана отношений против отношения медиан -- ЧЕСТНО, кто точнее",
        True,
        f"СКО мед.отн {rmse_p:.4f} против отн.медиан {rmse_m:.4f} -- "
        + (
            "медиана отношений точнее"
            if rmse_p < rmse_m
            else "РАЗНИЦЫ НЕТ на этой синтетике (выигрыш даёт ЧЕРЕДОВАНИЕ, см. Т1в)"
        ),
    )

    # --- Т2. ГЕЙТ СОСЕДА: замер с соседом НЕДЕЙСТВИТЕЛЕН, секундомер не запускался -------------
    calls = {"n": 0}

    def tim(fn, iters):
        calls["n"] += 1
        fn()
        return 1.0

    h = Harness(
        card=9,
        rounds=6,
        warmup=2,
        lock_mhz=None,
        smi=FakeSmi(foreign=[("12345", 312.0)]),
        timer=tim,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: True,
        label="сосед",
    )
    chk(
        "Т2 сосед на карте -> НЕДЕЙСТВИТЕЛЕН",
        not res.verdict.valid,
        (res.verdict.fatal[0][:70] + "...") if res.verdict.fatal else "",
    )
    chk(
        "Т2б при отказе секундомер НЕ ЗАПУСКАЛСЯ",
        calls["n"] == 0,
        f"вызовов таймера {calls['n']}",
    )

    # --- Т2в. Мощность по секундам ловит соседа БЕЗ чужого pid (MPS/чужой контейнер) -----------
    h = Harness(
        card=9,
        rounds=6,
        warmup=2,
        lock_mhz=None,
        smi=FakeSmi(power=180.0),
        timer=tim,
        idle_watts=70.0,
    )
    ok_env, v2, facts = h.precheck()
    chk(
        "Т2в мощность 180 Вт при пороге покоя 70 -> НЕДЕЙСТВИТЕЛЕН (utilization не спрашиваем)",
        not ok_env,
        v2.fatal[0][:80] if v2.fatal else "",
    )

    # --- Т3. ГЕЙТ КОРРЕКТНОСТИ ПЕРЕД ВРЕМЕНЕМ -------------------------------------------------
    calls["n"] = 0
    h = Harness(
        card=9,
        rounds=6,
        warmup=2,
        lock_mhz=None,
        smi=FakeSmi(),
        timer=tim,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: (False, "relL2 3.1e-1 -- вариант считает другое"),
        label="битая сверка",
    )
    chk(
        "Т3 сверка не прошла -> отказ мерить",
        not res.verdict.valid and calls["n"] == 0,
        f"вызовов таймера {calls['n']}",
    )
    calls["n"] = 0
    res = h.compare({"base": lambda: None, "var": lambda: None}, "base", check=None)
    chk(
        "Т3б сверки нет вовсе -> отказ мерить",
        not res.verdict.valid and calls["n"] == 0,
        res.verdict.fatal[0][:60] if res.verdict.fatal else "",
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=Harness.NO_CHECK,
        label="без сверки",
    )
    chk(
        "Т3в явный NO_CHECK -> мерит, но 'СВЕРКА НЕ ПРОВОДИЛАСЬ' в 'НЕ РАЗОБРАНО'",
        res.checked is False
        and any("СВЕРКА НЕ ПРОВОДИЛАСЬ" in u for u in res.verdict.unparsed),
    )

    # --- Т4. ПРОГРЕВ --------------------------------------------------------------------------
    seq = {"n": 0}

    def cold_timer(fn, iters):
        seq["n"] += 1
        i = (seq["n"] - 1) // 2
        return (2.0 if i == 0 else 1.0) * (1.0 + 0.002 * ((seq["n"] * 7919) % 5))

    h = Harness(
        card=9,
        rounds=8,
        warmup=0,
        lock_mhz=None,
        smi=FakeSmi(),
        timer=cold_timer,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: True,
        label="холодный первый раунд",
    )
    chk(
        "Т4 первый раунд выпал -> ПРОГРЕВ НЕ СОСТОЯЛСЯ -> недействителен",
        not res.verdict.valid and any("ПРОГРЕВ" in f for f in res.verdict.fatal),
    )
    h = Harness(
        card=9,
        rounds=8,
        warmup=0,
        lock_mhz=None,
        smi=FakeSmi(),
        timer=lambda fn, it: 1.0 + 0.001 * (seq.setdefault("k", 0) or 0),
        idle_watts=70.0,
    )
    res2 = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: True,
        label="ровный",
    )
    chk(
        "Т4б ровная серия -> прогрев признан состоявшимся",
        res2.summary["var"]["warmup_ok"] is True,
    )

    # --- Т5. ЧАСТОТЫ: дрейф ловится; снятие фиксации гарантировано -----------------------------
    h = Harness(
        card=9,
        rounds=6,
        warmup=1,
        lock_mhz=None,
        smi=FakeSmi(clk_seq=[1530, 1530, 1530, 307, 307, 1530] * 6),
        timer=lambda fn, it: 1.0,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: True,
        label="просадка частоты",
    )
    chk(
        "Т5 просадка 1530->307 МГц -> НЕДЕЙСТВИТЕЛЕН",
        not res.verdict.valid and any("ЧАСТОТА" in f for f in res.verdict.fatal),
    )

    log = []

    def fake_lock_runner(args):
        log.append(" ".join(args))
        return (0, "", "")

    lk = ClockLock(9, 1530, runner=fake_lock_runner)
    lk.acquire()

    def boom(fn, it):
        raise RuntimeError("ядро упало посреди раунда")

    h = Harness(
        card=9,
        rounds=4,
        warmup=0,
        lock_mhz=1530,
        smi=FakeSmi(),
        timer=boom,
        idle_watts=70.0,
    )
    h_lock_log = []
    try:
        orig = ClockLock._real
        ClockLock._real = lambda self, args: (
            h_lock_log.append(" ".join(args)),
            (0, "", ""),
        )[1]
        try:
            h.compare(
                {"base": lambda: None, "var": lambda: None}, "base", check=lambda: True
            )
        except RuntimeError:
            pass
    finally:
        ClockLock._real = orig
    chk(
        "Т5б падение посреди замера -> -rgc всё равно вызван",
        any("-rgc" in c for c in h_lock_log),
        " ; ".join(h_lock_log),
    )
    _release_all()
    chk(
        "Т5в аварийный выход (обработчик сигнала) снимает фиксацию",
        lk.released and lk.release_calls >= 1,
        f"вызовов release {lk.release_calls}",
    )
    log2 = []
    lk2 = ClockLock(
        9, 1530, runner=lambda a: (log2.append(" ".join(a)), (0, "", ""))[1]
    )
    lk2.acquire()
    lk2.release()
    lk2.release()
    chk(
        "Т5г повторное снятие идемпотентно (второй -rgc не уходит)",
        len([c for c in log2 if "-rgc" in c]) == 1,
        f"в журнале -rgc: {len([c for c in log2 if '-rgc' in c])}, "
        f"вызовов release {lk2.release_calls}",
    )

    # --- Т6. ЛОКАЛЬ И НЕРАЗОБРАННОЕ -----------------------------------------------------------
    up = []
    chk(
        "Т6 NBSP-разделитель тысяч ('5 238') разбирается в 5238",
        parse_num("5 238", up) == 5238.0,
        f"неразобранных {len(up)}",
    )
    chk(
        "Т6б русская дробная запятая ('51,11') -> 51.11",
        parse_num("51,11", up) == 51.11,
    )
    chk("Т6в английские тысячи ('1,530') -> 1530", parse_num("1,530", up) == 1530.0)
    up = []
    chk(
        "Т6г 'N/A' даёт None И запись в НЕ РАЗОБРАНО (а не 0)",
        parse_num("N/A", up, "power") is None and len(up) == 1,
        up[0] if up else "",
    )
    h = Harness(
        card=9,
        rounds=5,
        warmup=1,
        lock_mhz=None,
        smi=FakeSmi(broken_fields=True),
        timer=lambda fn, it: 1.0,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: True,
        label="битые поля",
    )
    chk(
        "Т6д непрочитанные поля карты попадают в 'НЕ РАЗОБРАНО'",
        len(res.verdict.unparsed) > 0,
        f"записей {len(res.verdict.unparsed)}",
    )

    # --- Т7. ТОЧКА ВХОДА ----------------------------------------------------------------------
    h = Harness(
        card=9,
        rounds=5,
        warmup=1,
        lock_mhz=None,
        smi=FakeSmi(),
        timer=lambda fn, it: 1.0,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None},
        "base",
        check=lambda: True,
        entry_probe={"base": lambda: "cutlass", "var": lambda: "cutlass"},
    )
    chk(
        "Т7 одинаковая точка входа у обоих вариантов -> A/B не состоялся",
        not res.verdict.valid and any("ТОЧКА ВХОДА" in f for f in res.verdict.fatal),
    )
    res = h.compare(
        {"base": lambda: None, "var": lambda: None}, "base", check=lambda: True
    )
    chk(
        "Т7б entry_probe не задан -> так и записано в 'НЕ РАЗОБРАНО'",
        any("точка входа НЕ ПОДТВЕРЖДЕНА" in u for u in res.verdict.unparsed),
    )

    # --- Т8. ПОЛОЖИТЕЛЬНЫЙ СКВОЗНОЙ ПРОГОН ----------------------------------------------------
    raw = _fake_workload(
        1.09, rounds=17, seed=7, drift_ramp=1.0, spike_p=0.0, noise=0.01
    )
    it = {"base": 0, "var": 0}

    def replay(fn, iters):
        nm = fn()
        i = it[nm]
        it[nm] += 1
        return raw[nm][min(i, len(raw[nm]) - 1)]

    h = Harness(
        card=9,
        rounds=17,
        warmup=3,
        lock_mhz=None,
        smi=FakeSmi(),
        timer=replay,
        idle_watts=70.0,
    )
    res = h.compare(
        {"base": lambda: "base", "var": lambda: "var"},
        "base",
        check=lambda: (True, ""),
        label="сквозной прогон",
        entry_probe={"base": lambda: "A", "var": lambda: "B"},
    )
    s = res.summary["var"]
    chk(
        "Т8 чистая среда -> ДЕЙСТВИТЕЛЕН и отношение 1.09 в ДИ",
        res.verdict.valid and s["ci_lo"] <= 1.09 <= s["ci_hi"],
        f"{s['ratio_median']:.4f} ДИ [{s['ci_lo']:.4f},{s['ci_hi']:.4f}]",
    )
    if verbose:
        print()
        print(res.report())

    n_ok = sum(1 for _, ok, _ in P if ok)
    if verbose:
        print(f"\nИТОГ САМОПРОВЕРКИ: {n_ok}/{len(P)} пройдено")
    return P


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="харнесс времени, который отказывается врать"
    )
    ap.add_argument(
        "--selftest", action="store_true", help="самопроверка на CPU, без железа"
    )
    ap.add_argument(
        "--precheck", action="store_true", help="только гейт среды (ничего не мерит)"
    )
    ap.add_argument(
        "--calibrate-idle",
        action="store_true",
        help="измерить порог покоя карты (запускать на ПУСТОЙ карте)",
    )
    ap.add_argument("--card", type=int, default=1, help="ФИЗИЧЕСКИЙ индекс карты")
    ap.add_argument("--idle-watts", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()

    if a.selftest:
        P = selftest()
        sys.exit(0 if all(ok for _, ok, _ in P) else 1)

    h = Harness(card=a.card, idle_watts=a.idle_watts, idle_probe_s=a.seconds)
    if a.calibrate_idle:
        d = h.calibrate_idle(seconds=max(10.0, a.seconds))
        print(json.dumps(d, indent=1, ensure_ascii=False))
        sys.exit(0 if d else 1)

    ok, v, facts = h.precheck()
    print(
        f"=== ГЕЙТ СРЕДЫ, карта {a.card}: "
        f"{'МОЖНО МЕРИТЬ' if ok else '*** МЕРИТЬ НЕЛЬЗЯ ***'} ==="
    )
    for k, val in facts.items():
        print(f"  {k}: {val}")
    for f in v.fatal:
        print(f"ОТМЕНА: {f}")
    for w in v.warn:
        print(f"замечание: {w}")
    print(f"--- НЕ РАЗОБРАНО / НЕ УСТАНОВЛЕНО ({len(v.unparsed)}) ---")
    for u in v.unparsed:
        print(f"  ? {u}")
    if not v.unparsed:
        print("  (пусто)")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

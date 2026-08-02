# -*- coding: utf-8 -*-
"""ПРОВЕРКА ОКРУЖЕНИЯ ПЕРЕД ТЯЖЁЛЫМ ЗАМЕРОМ. Секунды вместо потерянного прогона.

ЗАЧЕМ. 02.08.2026 три сквозных замера подряд сгорели НЕ на предмете, а на обвязке, и каждый раз
это выяснялось ПОСЛЕ загрузки 12-миллиардной модели:

  1. нет `accelerate` -> `device_map` падает (transformers требует его только ради device_map);
  2. нет `PYTHONPATH` к нашему пакету -> движок отвергает бэкенд: "fa2_sm70 kernels not importable";
  3. не хватило памяти: 12B fp16 ~24 ГБ ПЛЮС башни зрения и звука (+4 ГБ по прежнему замеру),
     и движок упал на выделении 240 МБ при 31.54 ГБ занятых.

Каждая проверяется за секунду ДО запуска. Ни одна не проверялась.

ЧТО ЭТО НЕ ДЕЛАЕТ. Не гарантирует, что замер осмыслен -- только что он вообще состоится. Годность
самого замера (форма, дисциплина, гейт корректности) -- отдельный разговор, см. `timeit.py`.

ЗАПУСК:
    python3 tools/precheck.py --python <интерпретатор> --import torch,vllm,fa2_sm70 \\
        --card 1 --model /путь/к/модели --need-ctx 32768 --pythonpath /путь/к/пакету
"""
import argparse
import json
import os
import subprocess
import sys

OK, BAD = "OK  ", "ОТКАЗ"


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=kw.pop("timeout", 120), **kw)


def check_imports(py, mods, pythonpath):
    """Импорты проверяются В ТОМ ЖЕ интерпретаторе, которым пойдёт замер, а не в текущем."""
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    out = []
    for m in mods:
        p = _run([py, "-c", f"import {m}; print(getattr({m},'__version__','?'))"], env=env)
        out.append((m, p.returncode == 0, (p.stdout or p.stderr).strip().splitlines()[-1][:80]
                    if (p.stdout or p.stderr).strip() else ""))
    return out


def model_bytes(path):
    """Размер весов ПО ЗАГОЛОВКАМ safetensors, а не по числу параметров.

    Число параметров врёт дважды: не знает dtype и не знает про башни зрения/звука, которые едут в
    память вместе с текстовой частью. Заголовок знает и то и другое.
    """
    total, per_prefix = 0, {}
    try:
        names = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    except OSError as e:
        return None, str(e), {}
    if not names:
        return None, "в каталоге нет .safetensors", {}
    for n in sorted(names):
        with open(os.path.join(path, n), "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            head = json.loads(f.read(hlen).decode("utf-8"))
        for k, v in head.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            total += b - a
            per_prefix[k.split(".")[0]] = per_prefix.get(k.split(".")[0], 0) + (b - a)
    return total, None, per_prefix


def build_env(bdir):
    """Проверки СБОРКИ. Добавлены 02.08.2026 после того, как один прогон приёмочного набора отказал
    ТРИЖДЫ подряд по трём разным причинам обвязки, и каждая стоила полного круга (~10 мин сборки):

      1. `_build/` принадлежит root -- след прогона профилировщика под `sudo` за 16 часов до этого;
      2. `CUDA_HOME` не задан -- фоновая оболочка не наследует его из профиля;
      3. (раньше, на сквозных) не тот интерпретатор / нет модуля / не хватило памяти.

    Ни одна не видна в отчёте как отказ обвязки: харнесс рапортует `22/23 FAIL`, и это читается как
    «правка сломала всё». Стоимость проверки -- миллисекунды.
    """
    out = []
    cu = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = os.path.join(cu, "bin", "nvcc") if cu else None
    out.append(("CUDA_HOME", bool(cu) and os.path.exists(nvcc or ""),
                f"{cu or '(не задан)'}" + ("" if not cu else f", nvcc {'есть' if os.path.exists(nvcc) else 'НЕТ'}")))
    if bdir:
        # Владельца проверяем ОТДЕЛЬНО от записи: каталог может быть чужим, но пока пустым и
        # формально записываемым -- а сломается он на первом же подкаталоге.
        ok, why = True, ""
        if os.path.exists(bdir):
            st = os.stat(bdir)
            if st.st_uid != os.getuid():
                ok, why = False, f"владелец uid={st.st_uid}, а мы uid={os.getuid()} (след sudo-прогона?)"
            elif not os.access(bdir, os.W_OK | os.X_OK):
                ok, why = False, "нет прав на запись"
            else:
                why = "свой, записываем"
        else:
            par = os.path.dirname(bdir.rstrip("/"))
            ok = os.access(par, os.W_OK)
            why = "не создан; родитель " + ("записываем" if ok else "НЕ записываем")
        out.append(("каталог сборки", ok, why))
    return out


_SMOKE_CU = r"""
#include <torch/extension.h>
__global__ void k(float* p) { p[threadIdx.x] = 1.0f; }
torch::Tensor f(torch::Tensor t) { k<<<1, 1>>>(t.data_ptr<float>()); return t; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("f", &f); }
"""


def smoke_build(py, pythonpath, arch):
    """ПРОБНАЯ СБОРКА ТЕМ ЖЕ ПУТЁМ, каким пойдёт замер. Единственная проверка здесь, которая не
    сверяет версии по отдельности, а ПРОХОДИТ цепочку целиком.

    Зачем именно так. 02.08.2026 приёмочный набор упал 22/23, и ни `CUDA_HOME`, ни права, ни импорт
    torch по отдельности отказа не показывали: набор пошёл БАЗОВЫМ conda-питоном вместо окружения
    `vllm`, оттуда `-ccbin` разрешился в conda-шный gcc старше 14, и nvcc отверг ХОСТ-компилятор.
    Такую связку не поймать проверкой полей -- её ловит только компиляция. Стоит ~20 с против ~10 мин
    сборки боевого файла и часа на диагностику.
    """
    import tempfile
    src = ("import torch, os, sys\n"
           "from torch.utils.cpp_extension import load\n"
           "d = sys.argv[1]\n"
           "open(os.path.join(d,'s.cu'),'w').write(%r)\n"
           "load(name='fa2_precheck_smoke', sources=[os.path.join(d,'s.cu')], build_directory=d,\n"
           "     extra_cuda_cflags=['-O0','-gencode=arch=compute_%s,code=sm_%s'], verbose=False)\n"
           "print('ok', torch.__version__, torch.version.cuda)\n" % (_SMOKE_CU, arch, arch))
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as d:
        p = _run([py, "-c", src, d], env=env, timeout=600)
    if p.returncode == 0:
        return True, (p.stdout or "").strip().splitlines()[-1][:90]
    err = ((p.stderr or "") + "\n" + (p.stdout or "")).strip()
    # Выдёргиваем ПЕРВУЮ содержательную строку отказа. Тонкость, стоившая одной итерации: наивное
    # "строка содержит error" цепляется за `raise CalledProcessError(...)` из трассы питона -- то
    # есть за ОБЁРТКУ отказа, а не за отказ. Настоящая причина у nvcc/gcc помечена `error:` или
    # `#error`; трассу питона отсеиваем явно. В конце стоит «ninja: build stopped», которое не
    # говорит ничего, поэтому запасной вариант -- не последняя строка, а первая непустая.
    lines = [l.strip() for l in err.splitlines() if l.strip()]
    py_noise = ("raise ", "File \"", "Traceback", "subprocess.", "  ")
    cand = [l for l in lines
            if ("error:" in l.lower() or "#error" in l.lower())
            and not l.lstrip().startswith(py_noise)]
    key = cand[0] if cand else (lines[0] if lines else "(пусто)")
    return False, key[:200]


def gpu_free(card):
    p = _run(["nvidia-smi", "-i", str(card), "--query-gpu=memory.total,memory.used,clocks.sm",
              "--format=csv,noheader,nounits"])
    if p.returncode != 0:
        return None
    tot, used, clk = (int(x.strip()) for x in p.stdout.strip().split(","))
    return tot, used, clk


def foreign(card):
    p = _run(["nvidia-smi", "-i", str(card), "--query-compute-apps=pid,used_memory",
              "--format=csv,noheader"])
    return [l for l in p.stdout.strip().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--import", dest="mods", default="", help="модули через запятую")
    ap.add_argument("--pythonpath", default="")
    ap.add_argument("--card", type=int, default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--need-ctx", type=int, default=0, help="длина контекста, под которую считать KV")
    # УМОЛЧАНИЯ ЗДЕСЬ НЕТ НАМЕРЕННО. Прежде стояло 16384 "для Gemma-4" -- и это оказался вклад
    # ТОЛЬКО глобальных слоёв (8 x 1 голова x 512 x 2 x 2 Б). Сорок скользящих добавляют своё, и
    # настоящая цена вышла в 6.5 раза выше: движок потребовал 3.32 ГиБ на 32832 токена там, где
    # проверка обещала 0.5. Инструмент печатал "ПРОВЕРЬ ЕЁ на своей модели" -- предупреждение,
    # которое никто не читает, потому что рядом стоит правдоподобное число. Теперь числа нет:
    # без явного значения оценка KV просто НЕ ДЕЛАЕТСЯ, и это честнее, чем делать её неверно.
    ap.add_argument("--kv-bytes-per-token", type=int, default=0,
                    help="байт KV на токен на ВСЕ слои. БЕЗ НЕГО оценка KV не делается. Считать по "
                         "СЛОЯМ: sum(n_kv_heads*head_dim)*2(K,V)*размер_элемента, послойно, а не по "
                         "одному типу слоя. Сверять с тем, что скажет сам движок при подъёме.")
    ap.add_argument("--build-dir", default="",
                    help="каталог JIT-сборки расширений (проверяются владелец и права)")
    ap.add_argument("--smoke-build", metavar="ARCH", nargs="?", const="70", default="",
                    help="пробная сборка крошечного .cu ТЕМ ЖЕ путём (по умолчанию sm_70); ~20 с")
    a = ap.parse_args()

    bad = []
    print(f"# интерпретатор: {a.python}")

    for name, ok, why in build_env(a.build_dir):
        print(f"{OK if ok else BAD}  {name:<20} {why}")
        if not ok:
            bad.append(f"{name}: {why}")

    if a.mods:
        for m, ok, ver in check_imports(a.python, [x for x in a.mods.split(",") if x], a.pythonpath):
            print(f"{OK if ok else BAD}  import {m:<16} {ver}")
            if not ok:
                bad.append(f"нет модуля {m} (в ЭТОМ интерпретаторе)")

    if a.smoke_build:
        ok, why = smoke_build(a.python, a.pythonpath, a.smoke_build)
        print(f"{OK if ok else BAD}  пробная сборка sm_{a.smoke_build}   {why}")
        if not ok:
            bad.append(f"пробная сборка не прошла: {why}")

    need = 0
    if a.model:
        b, err, per = model_bytes(a.model)
        if err:
            print(f"{BAD}  веса: {err}")
            bad.append(f"веса не прочитаны: {err}")
        else:
            print(f"{OK}  веса по заголовкам: {b/2**30:.1f} ГиБ")
            for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:6]:
                mark = "  <- БАЛЛАСТ для текстового замера" if any(
                    t in k for t in ("vision", "audio", "mm_", "multi_modal")) else ""
                print(f"        {k:<24} {v/2**30:6.2f} ГиБ{mark}")
            need += b

    if a.need_ctx and not a.kv_bytes_per_token:
        print(f"{BAD}  KV под {a.need_ctx} токенов НЕ ОЦЕНЕНА: не задан --kv-bytes-per-token")
        bad.append("оценка KV пропущена -- задайте --kv-bytes-per-token, посчитав ПОСЛОЙНО")
    if a.need_ctx and a.kv_bytes_per_token:
        kv = a.need_ctx * a.kv_bytes_per_token
        print(f"{OK}  KV под {a.need_ctx} токенов: {kv/2**30:.1f} ГиБ "
              f"(ставка {a.kv_bytes_per_token} Б/токен -- ПРОВЕРЬ ЕЁ на своей модели)")
        need += kv

    if a.card is not None:
        g = gpu_free(a.card)
        if g is None:
            print(f"{BAD}  карта {a.card} не опрашивается")
            bad.append("нет доступа к карте")
        else:
            tot, used, clk = g
            fr = foreign(a.card)
            print(f"{OK if not fr else BAD}  карта {a.card}: {used}/{tot} МиБ занято, {clk} МГц, "
                  f"чужих процессов {len(fr)}")
            for l in fr:
                print(f"        {l}")
            if fr:
                bad.append(f"на карте {a.card} чужие процессы -- замер ВРЕМЕНИ недействителен")
            if need:
                free_b = (tot - used) * 2**20
                # ЗАПАС 15 %: активации, фрагментация и профилировочный прогон движка. Именно на
                # профилировочном прогоне и случился отказ 02.08.2026.
                verdict = OK if free_b > need * 1.15 else BAD
                print(f"{verdict}  нужно ~{need/2**30:.1f} ГиБ + 15 % запаса, свободно "
                      f"{free_b/2**30:.1f} ГиБ")
                if free_b <= need * 1.15:
                    bad.append(f"памяти не хватит: нужно ~{need*1.15/2**30:.1f}, свободно "
                               f"{free_b/2**30:.1f} ГиБ")

    print()
    if bad:
        print("НЕ ЗАПУСКАТЬ. Отказы:")
        for b in bad:
            print(f"  * {b}")
        return 1
    print("ОКРУЖЕНИЕ ГОДНО. (Это не значит, что замер осмыслен -- только что он состоится.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

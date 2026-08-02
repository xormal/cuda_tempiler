#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EV_bits_table -- ТАБЛИЦА РАЗЛОЖЕНИЙ ТОЧНОСТИ ДЛЯ ВЕСОВ ЛИНЕЙНОЙ ЧАСТИ.

Строка таблицы = РАЗЛОЖЕНИЕ ВЕСА (сколько бит на вес, как хранятся масштабы, есть ли
компенсация ошибки). Колонки = ЦЕНА и УЩЕРБ:

  * БИТ НА ВЕС, включая накладные на масштабы/нули/индексы  -- это и есть цена;
  * предсказанное ВРЕМЯ слоя при M=32 и M=64 -- РАСЧЁТ (байты / достигнутая полоса из
    EV_mridge.md), НЕ замер;
  * АЛУ на элемент распаковки + укладывается ли она в ПРОСТОЙ счёта при M=32/64 -- РАСЧЁТ;
  * СКВОЗНОЙ ущерб flip_tf -- доля перевёрнутых токенов при принудительном контексте, ЗАМЕР.

Почему не relL2: ошибка восстановления произведения ОПРОВЕРГНУТА как страж
(E1_activation_entropy.md §5 -- ранжирует схемы наоборот). Она печатается справочно.

Почему веса, а не активации: при M=32 у gate,up веса 118 МБ против 245 КБ активаций
(EV_mridge.md). Активации -- 0.2 % трафика, их формат в этой таблице не рассматривается.

Подкоманды:
    selftest    -- проверки квантователей, разложения на плоскости, GPTQ и учёта бит (без модели)
    ref         -- сгенерировать fp16-эталон продолжений (один раз на модель)
    run         -- прогнать набор разложений и записать JSON со сквозным ущербом
    planes      -- замеры к ПРОГРЕССИВНОМУ ЧТЕНИЮ: побитовое равенство плоскостей,
                   цена второго прохода при M<=64, цена РАЗРЕЖЕННОГО чтения (нужна карта)
    table       -- собрать markdown-таблицу из JSON

Карты: только 0 и 1 (2 и 3 -- боевой сервер).
"""
import argparse
import json
import math
import os
import sys
import time

# tools/ содержит timeit.py, который перекрывает стандартный модуль при импорте torch.
# Убираем каталог скрипта из пути ДО импорта torch (иначе ImportError в torch._strobelight).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import torch

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------------------
# Крыши и доли полосы -- ЗАМЕРЕНЫ РАНЕЕ, здесь только используются.
# EV_mridge.md: доля ДОСТИЖИМОЙ полосы (841 ГБ/с), взятая cuBLAS-fp16 на боевых формах.
ROOF_GBS = 841.0
ROOF_TFLOPS = 93.6            # плотный HMMA fp16, замерено (EV_roof)
SM_COUNT = 80
FP32_LANES = 64               # на SM
CLOCK_HZ = 1.53e9
FP32_OPS_PER_S = SM_COUNT * FP32_LANES * CLOCK_HZ      # 7.83e12 однооперандных FP32/с
SLOTS_PER_S = SM_COUNT * 4 * CLOCK_HZ                  # 4.896e11 ВЫДАЧ варп-инструкций/с
ROOF_TFLOPS_ = 93.6
HMMA_MAC_PER_INSTR = ROOF_TFLOPS_ * 1e12 / 2 / SLOTS_PER_S   # ~95.6 MAC на выданную инструкцию

# доля достижимой полосы, взятая на форме при данном M (EV_mridge.md, замер)
BW_FRAC = {
    "q":       {32: 0.685, 64: 0.630},
    "k,v":     {32: 0.528, 64: 0.478},
    "o":       {32: 0.655, 64: 0.587},
    "gate,up": {32: 0.860, 64: 0.834},
    "down":    {32: 0.818, 64: 0.823},
}
# доля ТЕНЗОРНОГО счёта, взятая на форме при данном M (EV_mridge.md, замер).
COMPUTE_FRAC = {
    "q":       {32: 0.194, 64: 0.351},
    "k,v":     {32: 0.148, 64: 0.262},
    "o":       {32: 0.185, 64: 0.327},
    "gate,up": {32: 0.245, 64: 0.470},
    "down":    {32: 0.233, 64: 0.463},
}
# боевые формы Gemma-4-12B (K -> N на матрицу, одна пара на слой), из EV_mridge.py
GEMMA_SHAPES = [("q", 3840, 4096, 1), ("k,v", 3840, 2048, 2), ("o", 4096, 3840, 1),
                ("gate,up", 3840, 15360, 2), ("down", 15360, 3840, 1)]
GEMMA_LAYERS = 48
LLAMA_SHAPES = [("q", 2048, 2048, 1), ("k,v", 2048, 512, 2), ("o", 2048, 2048, 1),
                ("gate,up", 2048, 8192, 2), ("down", 8192, 2048, 1)]
LLAMA_LAYERS = 16

MODELS = {
    "gemma": dict(path="/mnt/d1/alex/models/gemma-4-12B-it-fix", vlm=True,
                  shapes=GEMMA_SHAPES, layers=GEMMA_LAYERS),
    "llama": dict(path="/mnt/d2/dnld/Llama-3.2-1B-Instruct", vlm=False,
                  shapes=LLAMA_SHAPES, layers=LLAMA_LAYERS),
}

CORPUS = "/mnt/d1/alex/reports/raw/corpus_wp.txt"


# =======================================================================================
#  1. КВАНТОВАТЕЛИ.  Группа идёт вдоль ВХОДНОЙ оси (как в GPTQ): W[out, in], группы по in.
# =======================================================================================
def _pow2(s):
    """Масштаб, округлённый до степени двойки ВВЕРХ: хранится один ПОКАЗАТЕЛЬ, распаковка =
    сложение показателей (АЛУ ~0), произведение код*масштаб в fp16 ТОЧНОЕ.
    Округление именно вверх: вниз -- обрезает диапазон группы и даёт клиппинг (проверено
    самотестом: max|dW| вырастал в 7 раз)."""
    return torch.exp2(torch.ceil(torch.log2(s.clamp_min(1e-30))))


def _nf_codebook(bits, device):
    """Неравномерный словарь: квантили нормального распределения (NF4-подобный).
    Хранение -- те же `bits` на вес, распаковка -- ТАБЛИЦА (АЛУ, не байты)."""
    n = 1 << bits
    try:
        from scipy.stats import norm  # не обязателен
        q = norm.ppf((torch.arange(n, dtype=torch.float64) + 0.5) / n).tolist()
    except Exception:
        # обратная функция ошибок через torch (без scipy)
        p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        q = (torch.erfinv(2 * p - 1) * math.sqrt(2)).tolist()
    v = torch.tensor(q, dtype=torch.float32, device=device)
    v = v / v.abs().max()
    return v


def _lloyd_codebook(x, bits, iters=25):
    """Словарь Ллойда-Макса, подогнанный под САМИ веса матрицы (нормированные)."""
    n = 1 << bits
    v = _nf_codebook(bits, x.device).clone()
    xs = x.flatten()
    if xs.numel() > 4_000_000:
        idx = torch.randint(0, xs.numel(), (4_000_000,), device=xs.device)
        xs = xs[idx]
    for _ in range(iters):
        b = torch.bucketize(xs, (v[1:] + v[:-1]) / 2)
        s = torch.zeros_like(v).scatter_add_(0, b, xs)
        c = torch.zeros_like(v).scatter_add_(0, b, torch.ones_like(xs))
        v = torch.where(c > 0, s / c.clamp_min(1), v)
        v, _ = torch.sort(v)
    return v


class Quant:
    """Одно разложение веса. group=0 -> одна группа на СТРОКУ (весь входной ряд)."""

    def __init__(self, bits=8, group=0, sym=True, scale_fmt="fp16", lut=None,
                 clip=False, outlier=0.0, plane_base=0, plane_mode="signed", plane_read="AB"):
        self.bits, self.group, self.sym = bits, group, sym
        self.scale_fmt, self.lut, self.clip, self.outlier = scale_fmt, lut, clip, outlier
        # ПЛОСКОСТИ: plane_base -- разрядов в плане A (байтово выровнен), остальные в плане B.
        # plane_mode='or' -- сборка (u_A<<drop)|u_B (план A в одиночку = УСЕЧЕНИЕ),
        # plane_mode='signed' -- (u_A<<drop)+знаковая поправка (план A = ОКРУГЛЕНИЕ).
        self.plane_base, self.plane_mode, self.plane_read = plane_base, plane_mode, plane_read
        self.plane_align = (1 << (bits - plane_base)) if (plane_base and not sym) else 1
        self.qmax = (1 << (bits - 1)) - 1 if sym else (1 << bits) - 1
        self._cb = None
        self.exp_span = []          # фактический разброс показателей (для честного счёта бит)

    # --- параметры одной группы: W [out, g] -> scale [out,1], zero [out,1] ------------
    def find_params(self, w):
        if self.lut is not None:
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
            z = torch.zeros_like(s)
        elif self.sym:
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / self.qmax
            z = torch.zeros_like(s)
        else:
            mx, mn = w.amax(1, keepdim=True), w.amin(1, keepdim=True)
            mx = torch.maximum(mx, torch.zeros_like(mx))
            mn = torch.minimum(mn, torch.zeros_like(mn))
            s = ((mx - mn) / self.qmax).clamp_min(1e-12)
            z = torch.round(-mn / s)
        if self.scale_fmt == "pow2":
            s = _pow2(s)
            if self.sym or self.lut is not None:
                pass
            else:
                z = torch.round(-w.amin(1, keepdim=True).clamp_max(0) / s).clamp(0, self.qmax)
        elif self.scale_fmt == "int8dq":
            # двойное квантование: масштабы группы -> int8 с одним fp16-надмасштабом на 256 групп
            pass       # применяется пакетно в quantize_matrix (нужен весь ряд масштабов)
        s = s.half().float()       # масштаб хранится в fp16
        if self.clip and self.lut is None:
            s = self._clip_search(w, s, z)
        if self.plane_align > 1:
            # ПЛОСКОСТИ: чтобы план A в одиночку был ЗАКОННЫМ int8, ноль обязан быть кратен 2^drop
            z = torch.round(z / self.plane_align) * self.plane_align
        self._s, self._z = s, z
        self._si, self._s1, self._z1 = 1.0 / s.squeeze(1), s.squeeze(1), z.squeeze(1)
        e = torch.log2(s.clamp_min(1e-30)).round()
        self.exp_span.append((float(e.min()), float(e.max())))
        return s, z

    def _clip_search(self, w, s, z):
        """Поиск ЛУЧШЕГО масштаба по MSE (data-free компенсация: стоит 0 байт и 0 ФЛОП в бою)."""
        best_s, best_e = s.clone(), None
        for f in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
            st = (s * f).clamp_min(1e-12)
            if self.scale_fmt == "pow2":
                st = _pow2(st)
            q = torch.clamp(torch.round(w / st) + z, self._lo(), self.qmax if not self.sym else self.qmax)
            e = ((q - z) * st - w).pow(2).sum(1, keepdim=True)
            if best_e is None:
                best_e = e
            else:
                m = e < best_e
                best_e = torch.where(m, e, best_e)
                best_s = torch.where(m, st, best_s)
        return best_s

    def _lo(self):
        return -self.qmax - 1 if self.sym else 0

    # --- квантовать один СТОЛБЕЦ (нужно GPTQ) ----------------------------------------
    def quantize_col(self, w):
        if self.lut is not None:
            cb = self._cb
            x = (w.unsqueeze(1) / self._s)
            b = torch.bucketize(x.flatten(), (cb[1:] + cb[:-1]) / 2)
            return (cb[b].view_as(x) * self._s).squeeze(1)
        q = torch.addcmul(self._z1, w, self._si).round_().clamp_(self._lo(), self.qmax)
        return q.sub_(self._z1).mul_(self._s1)

    # --- прямое квантование матрицы (RTN) --------------------------------------------
    def quantize_matrix(self, W):
        out, n = W.shape
        g = self.group if self.group else n
        if self.lut is not None:
            self._cb = _lloyd_codebook(W / W.abs().amax(1, keepdim=True).clamp_min(1e-12),
                                       self.bits) if self.lut == "lloyd" else _nf_codebook(self.bits, W.device)
        Q = torch.empty_like(W)
        scales = []
        for i in range(0, n, g):
            j = min(i + g, n)
            w = W[:, i:j]
            s, z = self.find_params(w)
            scales.append(s)
            if self.lut is not None:
                cb = self._cb
                x = w / s
                b = torch.bucketize(x.flatten(), (cb[1:] + cb[:-1]) / 2).view_as(x)
                Q[:, i:j] = cb[b] * s
            else:
                q = torch.clamp(torch.round(w / s) + z, self._lo(), self.qmax)
                if self.plane_base:
                    drop = self.bits - self.plane_base
                    if self.plane_read == "A":
                        # читаем ТОЛЬКО план A: 'or'-раскладка усекает, знаковая -- округляет
                        q = (torch.floor(q / (1 << drop)) if self.plane_mode == "or"
                             else torch.round(q / (1 << drop))) * (1 << drop)
                        q = q.clamp(self._lo(), self.qmax)
                Q[:, i:j] = (q - z) * s
        if self.scale_fmt == "int8dq" and len(scales) > 1:
            S = torch.cat(scales, 1)                       # [out, ngroups]
            sup = S.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
            S2 = torch.round(S / sup).clamp(0, 127) * sup
            Q = Q * (S2 / S.clamp_min(1e-30)).repeat_interleave(g, 1)[:, :n]
        return Q


# =======================================================================================
#  2. GPTQ -- КОМПЕНСАЦИЯ ОШИБКИ (offline; в бою стоит 0 байт и 0 ФЛОП)
# =======================================================================================
def gptq_quantize(W, H, quant, blocksize=128, percdamp=0.01):
    """W [out, in] fp32, H [in, in] fp32 = сумма x x^T по калибровке."""
    out, n = W.shape
    W = W.clone()
    dead = torch.diag(H) == 0
    H = H.clone()
    H[dead, dead] = 1.0
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H[range(n), range(n)] += damp
    L = torch.linalg.cholesky(H)
    del H                       # у больших n (down_proj: 15360^2 = 943 МБ) это критично
    Hinv = torch.cholesky_inverse(L)
    del L
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    g = quant.group if quant.group else n
    Q = torch.zeros_like(W)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % g == 0:
                quant.find_params(W[:, col:min(col + g, n)])
            w = W1[:, i]
            d = Hi[i, i]
            q = quant.quantize_col(w)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < n:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


# =======================================================================================
#  3. УЧЁТ БИТ НА ВЕС -- ЧЕСТНО, вместе со всеми накладными
# =======================================================================================
def packed_bits(b, pack):
    """Сколько бит РЕАЛЬНО занимает код при упаковке.
    'dense' -- плотно (нужна склейка через границу слова, +1-2 АЛУ/элемент);
    'word'  -- целое число кодов в 32-битном слове (распаковка одним сдвигом, но есть отход)."""
    if pack != "word":
        return float(b)
    return 32.0 / (32 // b)


def bits_per_weight(spec, K, N, exp_bits=5, pack="dense"):
    """K = входная размерность (ось группы), N = выходная. Возвращает (бит/вес, разбор)."""
    b = packed_bits(spec["bits"], pack)
    g = spec.get("group") or K
    parts = {"коды": float(b)}
    if spec.get("outlier", 0.0) > 0:
        p = spec["outlier"]
        # выбросные ВХОДНЫЕ каналы целиком в fp16 + список индексов (16 бит на канал)
        parts["коды"] = b * (1 - p)
        parts["выбросы fp16"] = 16.0 * p
        parts["индекс выбросов"] = 16.0 * p / N
    sf = spec.get("scale_fmt", "fp16")
    sb = {"fp16": 16.0, "pow2": float(exp_bits), "int8dq": 8.0 + 16.0 / 256}[sf]
    parts["масштабы"] = sb / g
    if not spec.get("sym", True):
        parts["нули"] = float(b) / g          # формат GPTQ: qzeros упакованы по b бит
    if spec.get("desc_act"):
        parts["g_idx"] = 32.0 / N
    if spec.get("lut"):
        parts["словарь"] = (1 << spec["bits"]) * 16.0 / (K * N)
    if spec.get("bias_corr"):
        parts["смещение-поправка"] = 16.0 / K     # fp16 на ВЫХОДНОЙ канал
    return sum(parts.values()), parts


def model_bytes(shapes, layers, spec, exp_bits=5, pack="dense"):
    """Байты линейной части модели при данном разложении (только веса)."""
    tot = 0.0
    for name, K, N, mult in shapes:
        bw, _ = bits_per_weight(spec, K, N, exp_bits, pack)
        tot += mult * K * N * bw / 8.0
    return tot * layers


def predict_layer_ms(shapes, spec, M, exp_bits=5, pack="dense", c_per_elem=None):
    """РАСЧЁТ (не замер): время ОДНОГО слоя.
    По каждой форме берётся МАКСИМУМ из двух ограничений:
      * байты / (доля полосы * 841 ГБ/с)  -- чтение;
      * (HMMA у считающих + распаковка у везущих) / 4.896e11 выдач/с -- ВЫДАЧА.
    Второе и есть ДНО лестницы: ниже него сжимать вес бессмысленно. Если c_per_elem=None,
    учитывается только чтение (чистая байтовая модель).
    Допущение, вынесенное явно: узкий вес читается с той же ДОЛЕЙ полосы, что fp16 (допущение
    EV_mridge.md; настоящее ядро W4A16 на Volta не построено)."""
    t = 0.0
    for name, K, N, mult in shapes:
        bw, _ = bits_per_weight(spec, K, N, exp_bits, pack)
        wb = mult * K * N * bw / 8.0
        ab = mult * 2.0 * (M * K + M * N)          # активации+выход в fp16
        gbs = BW_FRAC[name][M] * ROOF_GBS * 1e9
        t_read = (wb + ab) / gbs
        if c_per_elem is None:
            t += t_read
        else:
            nw = mult * K * N
            t_issue = nw * (M / HMMA_MAC_PER_INSTR + c_per_elem / 32.0) / SLOTS_PER_S
            t += max(t_read, t_issue)
    return t * 1e3


def alu_budget_per_weight(shapes, spec, M, exp_bits=5, pack="dense"):
    """Сколько ОДНООПЕРАНДНЫХ FP32/INT-операций на вес умещается в ПРОСТОЙ счёта.
    РАСЧЁТ: время слоя (байтовая модель) * 80 SM * 64 полосы * 1.53 ГГц / число весов.
    Тензорные ядра (HMMA) при этом заняты COMPUTE_FRAC, а FP32-конвейер Volta -- отдельный."""
    t = predict_layer_ms(shapes, spec, M, exp_bits, pack) * 1e-3
    nw = sum(mult * K * N for _, K, N, mult in shapes)
    return t * FP32_OPS_PER_S / nw


# =======================================================================================
#  4. МОДЕЛЬ, СХЕМА, СКВОЗНОЙ ЗАМЕР
# =======================================================================================
def sites_of(L):
    sa, mlp = L.self_attn, L.mlp
    return [("qkv", [getattr(sa, n, None) for n in ("q_proj", "k_proj", "v_proj")]),
            ("o", [getattr(sa, "o_proj", None)]),
            ("gate_up", [getattr(mlp, n, None) for n in ("gate_proj", "up_proj")]),
            ("down", [getattr(mlp, "down_proj", None)])]


def find_layers(model):
    import torch.nn as nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 4:
            if hasattr(mod[0], "self_attn") and hasattr(mod[0], "mlp"):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
    return best[1]


class Harness:
    def __init__(self, args):
        cfg = MODELS[args.model]
        self.args, self.cfg = args, cfg
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg["path"])
        if cfg["vlm"]:
            from transformers import AutoModelForImageTextToText as M
        else:
            from transformers import AutoModelForCausalLM as M
        try:                       # transformers >= 5 принимает dtype, старые -- torch_dtype
            self.model = M.from_pretrained(cfg["path"], dtype=torch.float16).to("cuda")
        except TypeError:
            self.model = M.from_pretrained(cfg["path"], torch_dtype=torch.float16).to("cuda")
        self.model.eval()
        self.layers = find_layers(self.model)
        print(f"# слоёв {len(self.layers)}", flush=True)
        # мастер-копия весов на CPU (восстановление между схемами)
        self.master = {}
        for li, L in enumerate(self.layers):
            for site, mods in sites_of(L):
                for k, m in enumerate(mods):
                    if m is not None:
                        # поправка-смещение добавляется НАМИ; своих смещений тут быть не должно
                        assert m.bias is None, f"у {site} слоя {li} уже есть смещение"
                        self.master[(li, site, k)] = m.weight.data.detach().to("cpu").clone()
        print(f"# мастер-копия: {sum(v.numel() for v in self.master.values())*2/2**30:.1f} ГиБ в ОЗУ",
              flush=True)
        txt = open(CORPUS, encoding="utf-8", errors="ignore").read()[args.skip:]
        half = len(txt) // 2
        self.cal_txt, self.eval_txt = txt[:half], txt[half:]

    def cut(self, t, n, ln):
        per = len(t) // n
        out = []
        for i in range(n):
            ids = self.tok(t[i * per:i * per + per], return_tensors="pt").input_ids[0]
            if ids.numel() >= ln:
                out.append(ids[:ln])
        return out

    def restore(self):
        for (li, site, k), w in self.master.items():
            mods = dict(sites_of(self.layers[li]))[site]
            mods[k].weight.data.copy_(w.to("cuda", non_blocking=True))
            mods[k].bias = None          # снять поправку-смещение прошлой схемы
        torch.cuda.synchronize()

    def gen(self, ids, n):
        past, cur, toks = None, ids.unsqueeze(0).to("cuda"), []
        for _ in range(n):
            o = self.model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            t = o.logits[0, -1].argmax()
            toks.append(int(t))
            cur = t.view(1, 1)
        return torch.tensor(toks)

    def flip_tf(self, prompts, ref):
        """Доля перевёрнутых токенов при ПРИНУДИТЕЛЬНОМ контексте (teacher forcing)."""
        bad = tot = 0
        per = []
        for p, r in zip(prompts, ref):
            full = torch.cat([p, r]).unsqueeze(0).to("cuda")
            lg = self.model(full).logits[0]
            pred = lg[len(p) - 1:-1].argmax(-1).cpu()
            bad += int((pred != r).sum())
            tot += r.numel()
            per.append(float((pred != r).float().mean()))
            del lg, full
        torch.cuda.empty_cache()
        return bad, tot, per


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------------------------------
#  Гессианы для GPTQ: полосами слоёв, прямой проход с ранним выходом.
# ---------------------------------------------------------------------------------------
class _Stop(Exception):
    pass


def collect_H(h, band, cal_chunks):
    """H[(layer,site)] = сумма x^T x по калибровке, для слоёв полосы `band`."""
    Hs, handles = {}, []

    def mk(key):
        def pre(mod, inp):
            # ПОЗИЦИЯ 0 ВЫБРАСЫВАЕТСЯ: на токене-стоке у Llama активация до 2474 медиан
            # (E1 §2). Один такой токен на 1024 доминирует в x^T x и разворачивает
            # компенсацию в минус -- проверено замером (см. отчёт, §"что опровергнуто").
            t = inp[0].detach()
            t = t[:, 1:] if t.dim() == 3 and t.shape[1] > 1 else t
            x = t.reshape(-1, t.shape[-1]).float()
            if key in Hs:
                Hs[key].addmm_(x.t(), x)
            else:
                Hs[key] = x.t() @ x
        return pre
    for li in band:
        for site, mods in sites_of(h.layers[li]):
            m = next((x for x in mods if x is not None), None)
            if m is not None:
                handles.append(m.register_forward_pre_hook(mk((li, site))))

    def stop(mod, inp, out):
        raise _Stop
    handles.append(h.layers[band[-1]].register_forward_hook(stop))
    for c in cal_chunks:
        try:
            h.model(c.unsqueeze(0).to("cuda"))
        except _Stop:
            pass
    for x in handles:
        x.remove()
    torch.cuda.empty_cache()
    return Hs


def mk_quant(s):
    return Quant(bits=s["bits"], group=s.get("group", 0), sym=s.get("sym", True),
                 scale_fmt=s.get("scale_fmt", "fp16"), lut=s.get("lut"),
                 clip=s.get("clip", False), plane_base=s.get("plane_base", 0),
                 plane_mode=s.get("plane_mode", "signed"), plane_read=s.get("plane_read", "AB"))


def _round_mantissa(y, mbits):
    """Округлить fp16-значение до mbits разрядов мантиссы (ОГРУБЛЕНИЕ СТЫКА)."""
    if mbits >= 10:
        return y
    i = y.view(torch.int16).to(torch.int32) & 0xFFFF
    drop = 10 - mbits
    add = 1 << (drop - 1)
    i = (i + add) & (~((1 << drop) - 1) & 0xFFFF)
    return (i.to(torch.int16)).view(torch.float16)


def apply_seam(h, spec):
    """Огрубить/уточнить СТЫК: округление выхода линейного слоя. Возвращает список хуков."""
    m = spec.get("seam")
    if not m:
        return []
    mb = {"m8": 8, "m6": 6, "m4": 4}[m]
    hs = []

    def hook(mod, inp, out):
        return _round_mantissa(out, mb)
    for L in h.layers:
        for site, mods in sites_of(L):
            for x in mods:
                if x is not None:
                    hs.append(x.register_forward_hook(hook))
    return hs


def channel_means(h, cal_chunks):
    """Средний выход КАЖДОГО линейного слоя по калибровке (вектор на выходной канал)."""
    acc, cnt, handles = {}, {}, []

    def mk(key):
        def hook(mod, inp, out):
            o = out.detach()
            o = o[:, 1:] if o.dim() == 3 and o.shape[1] > 1 else o
            o = o.reshape(-1, o.shape[-1]).float()
            if key in acc:
                acc[key] += o.sum(0)
                cnt[key] += o.shape[0]
            else:
                acc[key] = o.sum(0)
                cnt[key] = o.shape[0]
        return hook
    for li, L in enumerate(h.layers):
        for site, mods in sites_of(L):
            for k, m in enumerate(mods):
                if m is not None:
                    handles.append(m.register_forward_hook(mk((li, site, k))))
    for c in cal_chunks:
        h.model(c.unsqueeze(0).to("cuda"))
    for x in handles:
        x.remove()
    torch.cuda.empty_cache()
    return {k: acc[k] / cnt[k] for k in acc}


def input_means(h, cal_chunks):
    """Средний ВХОД каждой точки съёма по калибровке (вектор на входной канал)."""
    acc, cnt, handles = {}, {}, []

    def mk(key):
        def pre(mod, inp):
            t = inp[0].detach()
            t = t[:, 1:] if t.dim() == 3 and t.shape[1] > 1 else t   # без токена-стока
            x = t.reshape(-1, t.shape[-1]).float()
            if key in acc:
                acc[key] += x.sum(0)
                cnt[key] += x.shape[0]
            else:
                acc[key] = x.sum(0)
                cnt[key] = x.shape[0]
        return pre
    for li, L in enumerate(h.layers):
        for site, mods in sites_of(L):
            m = next((x for x in mods if x is not None), None)
            if m is not None:
                handles.append(m.register_forward_pre_hook(mk((li, site))))
    for c in cal_chunks:
        h.model(c.unsqueeze(0).to("cuda"))
    for x in handles:
        x.remove()
    torch.cuda.empty_cache()
    return {k: acc[k] / cnt[k] for k in acc}


def apply_bias_exact(h, mx):
    """ТОЧНАЯ офлайн-компенсация систематической ошибки ВЕСА через стык.
    Линейность даёт её без единого лишнего умножения матриц:
        E[W x] - E[Wq x] = (W - Wq) @ E[x],
    то есть достаточно СРЕДНЕГО ВХОДА. Поправка садится в смещение (в бою -- затравка
    аккумулятора, 0 байт на вес и 0 ФЛОП). Считается на ПЕРВОЙ половине корпуса, оценивается
    на ВТОРОЙ.
    ВАЖНО: компенсируется ТОЛЬКО ошибка веса этого слоя, а НЕ снос входа от предыдущих --
    сносовый вариант (`bias_corr='drift'`) замерен и оказался разрушительным (см. отчёт)."""
    import torch.nn as nn
    n = 0
    for li, L in enumerate(h.layers):
        for site, mods in sites_of(L):
            if (li, site) not in mx:
                continue
            m0 = mx[(li, site)]
            for k, m in enumerate(mods):
                if m is None or (li, site, k) not in h.master:
                    continue
                W0 = h.master[(li, site, k)].to("cuda", non_blocking=True).float()
                d = (W0 - m.weight.data.float()) @ m0
                del W0
                m.bias = nn.Parameter(d.half()) if m.bias is None else m.bias
                n += 1
    torch.cuda.empty_cache()
    return n


def apply_bias_correction(h, ref_means, cal_chunks):
    """ОФЛАЙН-КОМПЕНСАЦИЯ ЧЕРЕЗ СТЫК: систематическую составляющую ошибки выхода сворачиваем
    в СМЕЩЕНИЕ слоя (в бою -- начальное значение аккумулятора, 0 байт трафика на вес и 0 ФЛОП).
    Считается по КАЛИБРОВОЧНОЙ половине корпуса, оценивается на ВТОРОЙ -- разнос обязателен."""
    import torch.nn as nn
    q_means = channel_means(h, cal_chunks)
    n = 0
    for li, L in enumerate(h.layers):
        for site, mods in sites_of(L):
            for k, m in enumerate(mods):
                if m is None or (li, site, k) not in ref_means:
                    continue
                d = (ref_means[(li, site, k)] - q_means[(li, site, k)]).half()
                if m.bias is None:
                    m.bias = nn.Parameter(d.clone())
                else:
                    m.bias.data += d
                n += 1
    return n


def apply_scheme(h, spec, cal_chunks=None, verbose=True):
    """Наложить разложение на все выбранные матрицы (in place). Возвращает справочные метрики."""
    sites = spec.get("sites") or ["qkv", "o", "gate_up", "down"]
    lo, hi = spec.get("layers", (0, 10 ** 9))
    if spec.get("layer_frac"):
        f0, f1 = spec["layer_frac"]
        lo, hi = int(round(f0 * len(h.layers))), int(round(f1 * len(h.layers)))
    per_site_bits = spec.get("per_site_bits")           # {site: bits} для смешанного бюджета
    comp = spec.get("comp", "rtn")
    relL2, nrel = 0.0, 0
    t0 = time.time()
    todo = [li for li in range(len(h.layers)) if lo <= li < hi]

    def qspec_for(site):
        s = dict(spec)
        if per_site_bits and site in per_site_bits:
            s["bits"] = per_site_bits[site]
        return s

    if comp == "gptq":
        # ширина полосы по памяти под гессианы (у Gemma down_in даёт 15360^2*4 = 943 МБ на слой)
        band_budget = spec.get("band", 2 if len(h.layers) > 20 else 3)
        bands = [todo[i:i + band_budget] for i in range(0, len(todo), band_budget)]
        for band in bands:
            Hs = collect_H(h, band, cal_chunks)
            for li in band:
                for site, mods in sites_of(h.layers[li]):
                    if site not in sites:
                        continue
                    H = Hs.pop((li, site), None)
                    for m in mods:
                        if m is None:
                            continue
                        s = qspec_for(site)
                        q = mk_quant(s)
                        W = m.weight.data.float()
                        if q.lut is not None:
                            q._cb = (_lloyd_codebook(W / W.abs().amax(1, keepdim=True).clamp_min(1e-12), q.bits)
                                     if q.lut == "lloyd" else _nf_codebook(q.bits, W.device))
                        Wq = gptq_quantize(W, H, q)
                        relL2 += float((Wq - W).norm() / W.norm())
                        nrel += 1
                        m.weight.data.copy_(Wq.half())
                        del W, Wq
                torch.cuda.empty_cache()
            del Hs
            torch.cuda.empty_cache()
            if verbose:
                print(f"    полоса {band[0]}..{band[-1]} готова, {time.time()-t0:.0f} с", flush=True)
    else:
        for li in todo:
            for site, mods in sites_of(h.layers[li]):
                if site not in sites:
                    continue
                for m in mods:
                    if m is None:
                        continue
                    s = qspec_for(site)
                    q = mk_quant(s)
                    W = m.weight.data.float()
                    keep = None
                    if spec.get("outlier", 0.0) > 0:
                        # выбросные ВХОДНЫЕ каналы (столбцы W) целиком в fp16.
                        # отбор -- по вкладу столбца в ошибку: ||w_col|| * ширина столбца
                        score = W.abs().amax(0) * W.norm(dim=0)
                        k = max(1, int(round(spec["outlier"] * W.shape[1])))
                        idx = torch.topk(score, k).indices
                        keep = (idx, W[:, idx].clone())
                    Wq = q.quantize_matrix(W)
                    if keep is not None:
                        Wq[:, keep[0]] = keep[1]
                    relL2 += float((Wq - W).norm() / W.norm())
                    nrel += 1
                    m.weight.data.copy_(Wq.half())
                    del W, Wq
            torch.cuda.empty_cache()
    return {"relW": relL2 / max(nrel, 1), "matrices": nrel, "quant_s": round(time.time() - t0, 1)}


# =======================================================================================
#  5. НАБОРЫ РАЗЛОЖЕНИЙ
# =======================================================================================
def scheme_sets(names):
    """names -- одно имя набора или несколько через запятую (порядок сохраняется, дубли снимаются)."""
    if "," in names:
        out, seen = [], set()
        for n in names.split(","):
            for s in scheme_sets(n.strip()):
                if s["tag"] not in seen:
                    seen.add(s["tag"])
                    out.append(s)
        return out
    name = names
    S = []

    def add(tag, **kw):
        S.append(dict(tag=tag, **kw))

    if name in ("main", "all"):
        # порядок = порядок ВАЖНОСТИ: прогон возобновляем, и если он оборвётся, самое ценное
        # уже посчитано
        add("fp16", bits=16, group=0, none=True)
        add("int8-g128-asym-gptq [БОЕВАЯ]", bits=8, group=128, sym=False, comp="gptq")
        add("int8-row-sym-rtn [база E1]", bits=8, group=0, sym=True)
        add("int8-g128-asym-rtn", bits=8, group=128, sym=False)
        add("int4-g128-asym-rtn", bits=4, group=128, sym=False)
        add("int4-g128-asym-gptq", bits=4, group=128, sym=False, comp="gptq")
        add("int3-g128-asym-rtn", bits=3, group=128, sym=False)
        add("int4-g128-sym-rtn", bits=4, group=128, sym=True)
        add("int6-g128-asym-rtn", bits=6, group=128, sym=False)
        add("int5-g128-asym-rtn", bits=5, group=128, sym=False)
        # ДВЕ ПЛОСКОСТИ (план A = 8 бит байтово выровнен, план B = 2 бита):
        # ОДНО разложение с двумя уровнями добора, решение "брать ли B" -- на лету, по матрице.
        add("план A+B (int10; ноль кратен 4)", bits=10, group=128, sym=False, plane_base=8)
        add("план A один; знаковая поправка", bits=10, group=128, sym=False, plane_base=8,
            plane_read="A")
        add("план A один; ИЛИ-раскладка (усечение)", bits=10, group=128, sym=False,
            plane_base=8, plane_mode="or", plane_read="A")
        # --- ВЫШЕ int8: неиспользованные разряды мантиссы fp16 (0x6400|u даёт 10 бит) ---
        add("int10-g128-asym-rtn", bits=10, group=128, sym=False)
        add("int9-g128-asym-rtn", bits=9, group=128, sym=False)
        # int12 только СИММЕТРИЧНЫЙ: коды 0..4095 у асимметричного выходят за 2048 и уже НЕ
        # представимы в fp16 точно (см. selftest, граница целых)
        add("int12-g128-sym-rtn", bits=12, group=128, sym=True)
        # --- тяжёлые по АЛУ, лёгкие по байтам (курс 19:1 -- их цена в дешёвой валюте) ---
        add("nf4-g128-lut-rtn", bits=4, group=128, sym=True, lut="nf")
        add("lloyd4-g128-lut-rtn", bits=4, group=128, sym=True, lut="lloyd")
        add("lloyd3-g64-lut-rtn", bits=3, group=64, sym=True, lut="lloyd")
        add("int4-g64-asym-pow2-rtn", bits=4, group=64, sym=False, scale_fmt="pow2")
        add("int4-g32-asym-pow2-rtn", bits=4, group=32, sym=False, scale_fmt="pow2")
        add("int4-g64-asym-dq-rtn", bits=4, group=64, sym=False, scale_fmt="int8dq")
        # --- компенсация масштабом (data-free) -------------------------------------
        add("int4-g128-sym-clip-rtn", bits=4, group=128, sym=True, clip=True)
        add("int4-g128-asym-clip-rtn", bits=4, group=128, sym=False, clip=True)
        # --- смешанное: выбросные столбцы в fp16 ------------------------------------
        add("int4-g128-asym-rtn+0.5%fp16", bits=4, group=128, sym=False, outlier=0.005)
        add("int4-g128-asym-rtn+2%fp16", bits=4, group=128, sym=False, outlier=0.02)
        add("int3-g128-asym-rtn+2%fp16", bits=3, group=128, sym=False, outlier=0.02)
        # --- мельче группа ----------------------------------------------------------
        add("int4-g64-asym-rtn", bits=4, group=64, sym=False)
        add("int4-g32-asym-rtn", bits=4, group=32, sym=False)
        add("int8-g64-sym-rtn", bits=8, group=64, sym=True)
        add("int8-g32-sym-rtn", bits=8, group=32, sym=True)
        add("int8-g128-sym-rtn", bits=8, group=128, sym=True)
        add("int6-g64-asym-rtn", bits=6, group=64, sym=False)
        add("int5-g64-asym-rtn", bits=5, group=64, sym=False)
        # --- остальная компенсация (дорогие прогоны -- в конце) ---------------------
        add("int3-g128-asym-gptq", bits=3, group=128, sym=False, comp="gptq")
        add("int5-g128-asym-gptq", bits=5, group=128, sym=False, comp="gptq")
        add("int4-g64-asym-gptq", bits=4, group=64, sym=False, comp="gptq")
        # БОЕВОЙ РЕЦЕПТ КАК ЕСТЬ: bits=8 g128 asym desc_act=False, но слой 0 и q/k/v -- fp16
        add("БОЕВОЙ рецепт как в проде (L0 и qkv в fp16)", bits=8, group=128, sym=False,
            comp="gptq", sites=["o", "gate_up", "down"], layers=(1, 10 ** 9))
    if name in ("mods", "all"):
        # МОДИФИКАТОРЫ: сдвигают ли они таблицу целиком
        # 'drift' -- наивная свёртка ПОЛНОГО расхождения выхода (со сносом входа);
        # True -- ТОЧНАЯ свёртка ошибки веса (W-Wq)@E[x]
        add("МОД bias: int8-g128-asym-rtn", bits=8, group=128, sym=False, bias_corr="drift")
        add("МОД bias: int4-g128-asym-rtn", bits=4, group=128, sym=False, bias_corr="drift")
        add("МОД bias: int4-g128-asym-gptq", bits=4, group=128, sym=False, comp="gptq",
            bias_corr="drift")
        add("МОД bias: int3-g128-asym-rtn", bits=3, group=128, sym=False, bias_corr="drift")
    if name in ("bias2", "all"):
        add("МОД bias-точный: int8-g128-asym-rtn", bits=8, group=128, sym=False, bias_corr=True)
        add("МОД bias-точный: int4-g128-asym-rtn", bits=4, group=128, sym=False, bias_corr=True)
        add("МОД bias-точный: int3-g128-asym-rtn", bits=3, group=128, sym=False, bias_corr=True)
        add("МОД bias-точный: int6-g128-asym-rtn", bits=6, group=128, sym=False, bias_corr=True)
        # проверка МЕХАНИЗМА: у усечённого плана A ошибка СМЕЩЕНА по построению (полшага),
        # значит именно тут поправка обязана сработать -- если не сработает и здесь, приём мёртв
        add("МОД bias-точный: план A ИЛИ (усечение)", bits=10, group=128, sym=False,
            plane_base=8, plane_mode="or", plane_read="A", bias_corr=True)
        # СТЫК: огрубляем округление выхода линейного слоя (falsifier снизу)
        add("МОД стык m8: fp16 веса", bits=16, group=0, none=True, seam="m8")
        add("МОД стык m6: fp16 веса", bits=16, group=0, none=True, seam="m6")
        add("МОД стык m4: fp16 веса", bits=16, group=0, none=True, seam="m4")
        add("МОД стык m8: int4-g128-asym-rtn", bits=4, group=128, sym=False, seam="m8")
        add("МОД стык m6: int4-g128-asym-rtn", bits=4, group=128, sym=False, seam="m6")
    if name in ("sites", "all"):
        # ЧУВСТВИТЕЛЬНОСТЬ ПО МАТРИЦАМ: всё fp16, кроме одной точки в int3/int4
        for site in ("qkv", "o", "gate_up", "down"):
            add(f"only-{site}-int3-g128", bits=3, group=128, sym=False, sites=[site])
            add(f"only-{site}-int4-g128", bits=4, group=128, sym=False, sites=[site])
    if name in ("layers", "all"):
        # ЧУВСТВИТЕЛЬНОСТЬ ПО ГЛУБИНЕ: int3 в одной четверти слоёв, остальное fp16
        add("only-L0", bits=3, group=128, sym=False, layers=(0, 1))
        add("only-Q1", bits=3, group=128, sym=False, layer_frac=(0.0, 0.25))
        add("only-Q2", bits=3, group=128, sym=False, layer_frac=(0.25, 0.5))
        add("only-Q3", bits=3, group=128, sym=False, layer_frac=(0.5, 0.75))
        add("only-Q4", bits=3, group=128, sym=False, layer_frac=(0.75, 1.0))
        add("кроме-Q1 (Q1 в fp16)", bits=3, group=128, sym=False, layer_frac=(0.25, 1.0))
    if name in ("budget", "all"):
        # РАСПРЕДЕЛЁННЫЙ БЮДЖЕТ (то, что даёт прогрессивное чтение)
        add("budget-A: down8 rest4", bits=4, group=128, sym=False,
            per_site_bits={"down": 8})
        add("budget-B: qkv8 down8 rest4", bits=4, group=128, sym=False,
            per_site_bits={"down": 8, "qkv": 8})
        add("budget-C: gate_up4 rest8", bits=8, group=128, sym=False,
            per_site_bits={"gate_up": 4})
        add("budget-D: down5 rest4", bits=4, group=128, sym=False,
            per_site_bits={"down": 5})
    if name in ("budget2", "all"):
        # ПРОВЕРКА ПРАВИЛА из §6.2: сдвиги на ОДИН бит по формуле  db = log2(D_i/w_i) - среднее
        add("budget-E: gate_up5 rest6", bits=6, group=128, sym=False,
            per_site_bits={"gate_up": 5})
        add("budget-F: o8 qkv7 down6 gate_up5", bits=6, group=128, sym=False,
            per_site_bits={"o": 8, "qkv": 7, "down": 6, "gate_up": 5})
        add("budget-G: o7 qkv6 down5 gate_up4", bits=5, group=128, sym=False,
            per_site_bits={"o": 7, "qkv": 6, "down": 5, "gate_up": 4})
    return S


# =======================================================================================
#  6. ПОДКОМАНДЫ
# =======================================================================================
def cmd_ref(args):
    h = Harness(args)
    prompts = h.cut(h.eval_txt, args.prompts, args.plen)
    ref = [h.gen(p, args.gen).tolist() for p in prompts]
    json.dump({"prompts": [p.tolist() for p in prompts], "ref": ref, "plen": args.plen,
               "gen": args.gen, "model": h.cfg["path"]}, open(args.ref, "w"))
    print("эталон записан:", args.ref, "позиций", sum(len(r) for r in ref))


def cmd_run(args):
    h = Harness(args)
    R = json.load(open(args.ref))
    prompts = [torch.tensor(p) for p in R["prompts"]]
    ref = [torch.tensor(r) for r in R["ref"]]
    cal_chunks = h.cut(h.cal_txt, args.cal, args.callen)
    print(f"# калибровка: {len(cal_chunks)} x {args.callen} = "
          f"{len(cal_chunks)*args.callen} токенов", flush=True)
    done = {}
    if os.path.exists(args.out) and args.resume:
        done = {r["tag"]: r for r in json.load(open(args.out))}
    schemes = [s for s in scheme_sets(args.set) if not args.only or s["tag"] in args.only.split(",")]
    rows = list(done.values())
    for spec in schemes:
        tag = spec["tag"]
        if tag in done:
            print("пропуск (есть):", tag, flush=True)
            continue
        print(f"=== {tag} ===", flush=True)
        t0 = time.time()
        h.restore()
        info = {}
        ref_means = None
        if spec.get("bias_corr") == "drift":
            ref_means = channel_means(h, cal_chunks)      # эталонные средние на fp16-весах
        if not spec.get("none"):
            info = apply_scheme(h, spec, cal_chunks)
        if spec.get("bias_corr") == "drift":
            info["bias_fixed"] = apply_bias_correction(h, ref_means, cal_chunks)
            del ref_means
            torch.cuda.empty_cache()
        elif spec.get("bias_corr"):
            info["bias_fixed"] = apply_bias_exact(h, input_means(h, cal_chunks))
        seam_hooks = apply_seam(h, spec)
        bad, tot, per = h.flip_tf(prompts, ref)
        for x in seam_hooks:
            x.remove()
        lo, hi = wilson(bad, tot)
        row = dict(spec)
        row.pop("none", None)
        row.update(flip_tf=bad / tot, flip_lo=lo, flip_hi=hi, n=tot, per_prompt=per,
                   secs=round(time.time() - t0, 1), **info)
        print(f"    flip_tf = {100*bad/tot:.2f} %  [{100*lo:.2f}; {100*hi:.2f}]  "
              f"N={tot}  ({row['secs']} с)", flush=True)
        rows.append(row)
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("записано:", args.out)


# ---------------------------------------------------------------------------------------
def cmd_planes(args):
    """Замеры к ПРОГРЕССИВНОМУ ЧТЕНИЮ (нужна свободная карта)."""
    dev = "cuda"
    out = {}
    # --- (а) ГЕЙТ КОРРЕКТНОСТИ: 4+4 плоскости == int8 ПОБИТОВО --------------------
    W = torch.randint(-128, 128, (4096, 4096), device=dev, dtype=torch.int32)
    hi = (W >> 4)                       # арифметический сдвиг: старшая плоскость со знаком
    lo = (W & 0xF)                      # младшая плоскость без знака
    rec = (hi << 4) | lo
    out["planes_bitexact"] = bool(torch.equal(rec, W))
    # то же в fp16-операндах тензорного ядра
    A = torch.randn(64, 4096, device=dev, dtype=torch.float16)
    Wf = W.half()
    y1 = (A @ Wf.T.contiguous().T.float()) if False else (A.float() @ W.float())
    y2 = 16.0 * (A.float() @ hi.float()) + (A.float() @ lo.float())
    out["planes_gemm_rel_norm"] = float((y1 - y2).norm() / y1.norm())
    yh = 16.0 * (A @ hi.half()).float() + (A @ lo.half()).float()
    out["planes_gemm_fp16op_rel_norm"] = float((y1 - yh).norm() / y1.norm())
    # ИСЧЕРПЫВАЮЩАЯ сверка обратимости int10 (все 1024 значения, не выборка):
    # u10 -> (планA=u10>>2, планB=u10&3) -> сборка -> 0x6400|u10 -> значение 1024+u10
    u = torch.arange(1024, device=dev, dtype=torch.int32)
    pa, pb = u >> 2, u & 3
    back = (pa << 2) | pb
    val = (0x6400 | back).to(torch.int16).view(torch.float16).float()
    out["int10_planes_roundtrip_all1024"] = bool(torch.equal(back, u)) and \
        bool(torch.equal(val, (1024 + u).float()))
    # то же для ЗНАКОВОЙ поправки: планA = round(u/4) (сам по себе ОКРУГЛЁННЫЙ int8),
    # планB = u - 4*планA in {-2,-1,0,1}. Плата: коды 1022 и 1023 недостижимы (планA<=255),
    # то есть размах теряет 0.2 % -- это и есть вся цена того, что план A стал округлением.
    u2 = torch.arange(1022, device=dev, dtype=torch.int32)
    pa2 = torch.div(u2 + 2, 4, rounding_mode="floor").clamp(0, 255)
    pb2 = u2 - 4 * pa2
    out["int10_signed_split_ok_0_1021"] = bool(torch.equal(4 * pa2 + pb2, u2)) and \
        bool(pb2.min() >= -2 and pb2.max() <= 1)
    out["int10_signed_split_lost_codes"] = 2
    del W, hi, lo, rec, A, Wf, y1, y2, yh
    torch.cuda.empty_cache()

    def bench(fn, it=50, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / it

    # --- (б) ЦЕНА ВТОРОГО ПРОХОДА при M<=64 --------------------------------------
    # эквивалент по ТРАФИКУ: чтение b-битного веса [K,N] = чтение fp16-веса [K*b/16, N].
    # int8 одним проходом  <->  fp16 [K/2, N]; 4+4 двумя проходами <-> два fp16 [K/4, N].
    K, N = 3840, 15360         # gate,up у Gemma-4
    res = {}
    for M in (32, 64):
        a2 = torch.randn(M, K // 2, device=dev, dtype=torch.float16)
        w2 = torch.randn(K // 2, N, device=dev, dtype=torch.float16)
        a4 = torch.randn(M, K // 4, device=dev, dtype=torch.float16)
        w4a = torch.randn(K // 4, N, device=dev, dtype=torch.float16)
        w4b = torch.randn(K // 4, N, device=dev, dtype=torch.float16)
        a1 = torch.randn(M, K, device=dev, dtype=torch.float16)
        w1 = torch.randn(K, N, device=dev, dtype=torch.float16)
        t_fp16 = bench(lambda: torch.matmul(a1, w1))
        t_one = bench(lambda: torch.matmul(a2, w2))
        t_two = bench(lambda: torch.matmul(a4, w4a) + torch.matmul(a4, w4b))
        t_base = bench(lambda: torch.matmul(a4, w4a))
        res[M] = dict(fp16_ms=t_fp16 * 1e3, int8_1pass_ms=t_one * 1e3,
                      planes_4p4_2pass_ms=t_two * 1e3, base4_only_ms=t_base * 1e3,
                      second_pass_overhead=t_two / t_one - 1.0,
                      refine_cost_vs_base=t_two / t_base - 1.0)
        del a2, w2, a4, w4a, w4b, a1, w1
        torch.cuda.empty_cache()
    out["second_pass"] = res

    # --- (в) ЦЕНА НЕСПЛОШНОГО ЧТЕНИЯ уточняющей плоскости -------------------------
    # Читаем ПОЛОВИНУ байт большого тензора кусками разной длины -- РЕЗКОЙ, а не выборкой по
    # индексу (индекс сам создаёт трафик 8 Б на элемент и меряет не то: проверено, все
    # гранулярности слипались в 89 ГБ/с).
    nbytes = 512 << 20
    buf = torch.empty(nbytes // 2, device=dev, dtype=torch.float16)
    buf.uniform_(-1, 1)
    gran = {}
    for chunk_b in (32, 64, 128, 256, 512, 2048, 16384, 262144):
        el = chunk_b // 2
        v = buf.view(-1, 2 * el)[:, :el]                   # каждый второй кусок длиной chunk_b
        got = v.numel() * 2
        t = bench(lambda: v.sum(), it=20, warm=5)
        gran[chunk_b] = dict(ms=t * 1e3, gbs_payload=got / t / 1e9)
    t = bench(lambda: buf[:buf.numel() // 2].sum(), it=20, warm=5)
    gran["сплошной"] = dict(ms=t * 1e3, gbs_payload=(buf.numel() // 2) * 2 / t / 1e9)
    out["granularity"] = gran

    # --- (г) ДВА СЛИТНЫХ ПОТОКА 4:1 против ОДНОГО того же объёма -------------------
    # план A и план B как ДВА выровненных потока -- условие (б) пятого дополнения.
    n = 256 << 20
    one = torch.empty(n, device=dev, dtype=torch.int8).random_(-100, 100)
    a = torch.empty(n * 4 // 5, device=dev, dtype=torch.int8).random_(-100, 100)
    b = torch.empty(n // 5, device=dev, dtype=torch.int8).random_(-100, 100)
    t1 = bench(lambda: one.sum(dtype=torch.int32), it=20, warm=5)
    t2 = bench(lambda: a.sum(dtype=torch.int32) + b.sum(dtype=torch.int32), it=20, warm=5)
    out["two_streams"] = dict(one_ms=t1 * 1e3, two_ms=t2 * 1e3,
                              gbs_one=n / t1 / 1e9, gbs_two=n / t2 / 1e9,
                              overhead=t2 / t1 - 1.0)
    del one, a, b, buf
    torch.cuda.empty_cache()
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------------------
def cmd_selftest(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok ' if cond else 'FAIL'}] {name} {extra}")

    # 1. учёт бит
    b, parts = bits_per_weight(dict(bits=4, group=128, sym=False), 3840, 15360)
    chk("бит/вес int4-g128-asym = 4.15625", abs(b - (4 + 16 / 128 + 4 / 128)) < 1e-9, f"{b}")
    b2, _ = bits_per_weight(dict(bits=8, group=0, sym=True), 3840, 15360)
    chk("бит/вес int8-row-sym = 8+16/3840", abs(b2 - (8 + 16 / 3840)) < 1e-9, f"{b2:.5f}")
    b3, _ = bits_per_weight(dict(bits=4, group=64, sym=False, scale_fmt="pow2"), 3840, 15360)
    chk("бит/вес int4-g64-pow2 < int4-g64-fp16",
        b3 < bits_per_weight(dict(bits=4, group=64, sym=False), 3840, 15360)[0], f"{b3:.4f}")

    # 2. квантователи: ошибка не больше половины шага
    W = torch.randn(256, 512, device=dev)
    for spec in (dict(bits=8, group=0, sym=True), dict(bits=4, group=128, sym=False),
                 dict(bits=4, group=64, sym=False, scale_fmt="pow2"),
                 dict(bits=8, group=32, sym=True, scale_fmt="int8dq")):
        q = Quant(**spec)
        Q = q.quantize_matrix(W)
        # шаг задан САМОЙ ШИРОКОЙ группой; у pow2 масштаб округлён вверх -> шаг до 2x
        step = W.abs().amax() / (1 << (spec["bits"] - 1))
        tol = 2.1 if spec.get("scale_fmt") == "pow2" else 1.05
        chk(f"RTN {spec} ошибка <= шаг", float((Q - W).abs().max()) <= float(step) * tol,
            f"max|dW|={float((Q-W).abs().max()):.5f}")
    # мельче группа -> не хуже
    e = {}
    for g in (0, 128, 64, 32):
        e[g] = float((Quant(bits=4, group=g, sym=False).quantize_matrix(W) - W).norm())
    chk("мельче группа -> меньше ошибка", e[0] > e[128] > e[64] > e[32], str({k: round(v, 3) for k, v in e.items()}))

    # 3. плоскости 4+4 == int8 ПОБИТОВО
    Wi = torch.randint(-128, 128, (512, 512), device=dev, dtype=torch.int32)
    rec = ((Wi >> 4) << 4) | (Wi & 0xF)
    chk("4+4 плоскости == int8 побитово", bool(torch.equal(rec, Wi)))
    A = torch.randn(32, 512, device=dev, dtype=torch.float16)
    # ГЕЙТ: у HMMA.884 аккумулятор fp32, поэтому оба прохода складываются в ОДИН fp32-накопитель.
    # Эмулируем это fp32-умножением (операнды точно представимы в fp16 -> произведения точные).
    y1 = A.float() @ Wi.float()
    y2 = 16.0 * (A.float() @ (Wi >> 4).float()) + (A.float() @ (Wi & 0xF).float())
    r = float((y1 - y2).norm() / y1.norm())
    chk("два прохода по плоскостям == один по int8 (аккумулятор fp32)", r < 1e-6,
        f"rel норма={r:.2e}")
    # справочно: если результат КАЖДОГО прохода округлять до fp16 (накопитель НЕ разделяется),
    # точность падает -- это и есть требование к реализации.
    y3 = 16.0 * (A @ (Wi >> 4).half()).float() + (A @ (Wi & 0xF).half()).float()
    print(f"       [справочно] с округлением каждого прохода до fp16: "
          f"rel норма={float((y1-y3).norm()/y1.norm()):.2e}")

    # 4. GPTQ не хуже RTN на своей же цели ||(W-Wq)X||
    torch.manual_seed(0)
    Wt = torch.randn(128, 256, device=dev)
    X = torch.randn(4096, 256, device=dev)
    H = X.t() @ X
    q = Quant(bits=3, group=64, sym=False)
    Qr = q.quantize_matrix(Wt)
    Qg = gptq_quantize(Wt, H, Quant(bits=3, group=64, sym=False))
    er = float(((Wt - Qr) @ X.t()).norm())
    eg = float(((Wt - Qg) @ X.t()).norm())
    chk("GPTQ <= RTN по взвешенной ошибке", eg < er, f"rtn={er:.1f} gptq={eg:.1f} ({eg/er:.3f}x)")

    # 5. словарь Ллойда не хуже равномерного на гауссе
    g = torch.randn(64, 4096, device=dev)
    eu = float((Quant(bits=4, group=128, sym=True).quantize_matrix(g) - g).norm())
    el = float((Quant(bits=4, group=128, sym=True, lut="lloyd").quantize_matrix(g) - g).norm())
    chk("LUT(Ллойд) <= равномерного на гауссовом весе", el < eu, f"uni={eu:.1f} lut={el:.1f}")

    # 5b. ГРАНИЦА ЦЕЛЫХ В fp16 и приём 0x6400|u (сколько разрядов даёт ОДИН OR) -- ЗАМЕР
    u = torch.arange(0, 1024, device=dev, dtype=torch.int32)
    pat = (0x6400 | u).to(torch.int16).view(torch.float16).float()
    chk("0x6400|u == 1024+u для всех u в 0..1023 (=> ОДИН OR даёт 10 бит)",
        bool(torch.equal(pat, (1024 + u).float())))
    big = torch.arange(2040, 2060, device=dev, dtype=torch.float32)
    exact = (big.half().float() == big)
    first_bad = int(big[~exact].min())      # выше 2048 точны только ЧЁТНЫЕ, шаг стал 2
    chk("fp16 точен по ВСЕМ целым до 2048, 2049 уже нет", first_bad == 2049,
        f"первое неточное {first_bad}, сплошной диапазон до {first_bad-1}")
    # значит: sym-коды до ±2048 (int12) точны как ОПЕРАНД, asym-коды 0..2^b-1 -- до int11,
    # а приём 0x6400|u (без конверсии) -- до int10.
    chk("asym int12 (код 4095) НЕ представим точно", float(torch.tensor(4095.0).half()) != 4095.0
        or float(torch.tensor(4095.0).half()) == 4096.0, f"{float(torch.tensor(4095.0).half())}")

    # 6. доверительный интервал
    lo, hi = wilson(50, 1000)
    chk("интервал Вилсона накрывает долю", lo < 0.05 < hi, f"[{lo:.4f};{hi:.4f}]")

    # 7. предсказание времени монотонно по битам
    t8 = predict_layer_ms(GEMMA_SHAPES, dict(bits=8, group=128, sym=False), 32)
    t4 = predict_layer_ms(GEMMA_SHAPES, dict(bits=4, group=128, sym=False), 32)
    t16 = predict_layer_ms(GEMMA_SHAPES, dict(bits=16, group=0), 32)
    chk("время: fp16 > int8 > int4", t16 > t8 > t4, f"{t16:.3f} {t8:.3f} {t4:.3f} мс")
    print("\nИТОГ:", "ВСЁ ПРОШЛО" if ok else "ЕСТЬ ПАДЕНИЯ")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------------------
# --- ЗАМЕРЕННАЯ стоимость РАСПАКОВКИ (инструкций на элемент, разностный SASS, sm_70) -------
# tools/unpack_sass.cu; в числе учтены и LDG, и STS -- это ПОЛНАЯ работа везущего варпа.
SASS_PER_ELEM = {
    "fp16": 0.750, "int8": 1.000, "int8_pow2": 0.750, "int4": 1.750, "int5": 2.750,
    "int6": 2.500, "int3": 2.750, "int9": 3.250, "int10_prog": 3.750, "int10_byte": 3.250,
    "lut4": 3.750,
}
def sass_key(spec):
    b = spec.get("bits", 16)
    if spec.get("lut"):
        return "lut4"
    if spec.get("plane_base"):
        return "int10_prog"
    if b >= 16:
        return "fp16"
    if b == 8:
        return "int8_pow2" if spec.get("scale_fmt") == "pow2" else "int8"
    return {3: "int3", 4: "int4", 5: "int5", 6: "int6", 9: "int9", 10: "int10_prog",
            12: "int6"}.get(b, "int8")


def floor_bits(shape, M, c_per_elem):
    """ДНО ЛЕСТНИЦЫ для одной формы: ниже этого числа бит время перестаёт падать, потому что
    выдача (HMMA у считающих + распаковка у везущих) съедает весь бюджет тактов.
    Вывод: slots/элемент = 4.896e11 * (b/8)/BW_eff должно покрывать M/95.6 + c/32."""
    bw_eff = BW_FRAC[shape][M] * ROOF_GBS * 1e9
    per_bit = SLOTS_PER_S / (8 * bw_eff)
    return (M / HMMA_MAC_PER_INSTR + c_per_elem / 32.0) / per_bit


SHAPE_SITE = {"q": "qkv", "k,v": "qkv", "o": "o", "gate,up": "gate_up", "down": "down"}


def mix_bits(r, spec, name, K, N, pack="dense", nlayers=None):
    """Честные бит/вес для строки, которая квантует НЕ ВСЁ: часть точек и часть слоёв остаются
    fp16. Возвращает (бит/вес, доля квантованного)."""
    site = SHAPE_SITE[name]
    sites = r.get("sites") or ["qkv", "o", "gate_up", "down"]
    b16 = 16.0 + 0.0
    if site not in sites:
        return b16, 0.0
    s = dict(spec)
    if r.get("per_site_bits", {}).get(site):
        s["bits"] = r["per_site_bits"][site]
    bq, _ = bits_per_weight(s, K, N, pack=pack)
    n = nlayers or 48
    if r.get("layer_frac"):
        f = r["layer_frac"][1] - r["layer_frac"][0]
    elif r.get("layers"):
        lo, hi = r["layers"]
        f = (min(hi, n) - lo) / n
    else:
        f = 1.0
    return f * bq + (1 - f) * b16, f


def mix_layer_ms(r, spec, shapes, M, c):
    """То же, что predict_layer_ms, но с учётом частично квантованных строк."""
    t = 0.0
    for name, K, N, mult in shapes:
        bw, f = mix_bits(r, spec, name, K, N)
        wb = mult * K * N * bw / 8.0
        ab = mult * 2.0 * (M * K + M * N)
        t_read = (wb + ab) / (BW_FRAC[name][M] * ROOF_GBS * 1e9)
        cc = f * c + (1 - f) * SASS_PER_ELEM["fp16"]
        t_issue = mult * K * N * (M / HMMA_MAC_PER_INSTR + cc / 32.0) / SLOTS_PER_S
        t += max(t_read, t_issue)
    return t * 1e3


def spec_of(r):
    s = {k: r[k] for k in ("bits", "group", "sym", "scale_fmt", "lut", "outlier", "desc_act",
                           "plane_base", "bias_corr") if k in r}
    s.setdefault("bits", 16)
    return s


def cmd_table(args):
    shapes = MODELS[args.model]["shapes"]
    layers = MODELS[args.model]["layers"]
    rows = json.load(open(args.out))
    nw = sum(m * K * N for _, K, N, m in shapes)
    t16_32 = predict_layer_ms(shapes, dict(bits=16, group=0), 32, c_per_elem=0.75)
    t16_64 = predict_layer_ms(shapes, dict(bits=16, group=0), 64, c_per_elem=0.75)
    groups = {"ОСНОВНАЯ ТАБЛИЦА": [], "ЧУВСТВИТЕЛЬНОСТЬ (что во что кладём)": [],
              "РАСПРЕДЕЛЁННЫЙ БЮДЖЕТ": [], "МОДИФИКАТОРЫ": []}
    for r in rows:
        t = r["tag"]
        k = ("МОДИФИКАТОРЫ" if t.startswith("МОД") else
             "РАСПРЕДЕЛЁННЫЙ БЮДЖЕТ" if t.startswith("budget") else
             "ЧУВСТВИТЕЛЬНОСТЬ (что во что кладём)" if t.startswith(("only-", "кроме-")) else
             "ОСНОВНАЯ ТАБЛИЦА")
        groups[k].append(r)
    for gname, grows in groups.items():
        if not grows:
            continue
        print(f"\n### {gname}\n")
        _print_rows(grows, shapes, layers, nw, t16_32, t16_64)
    print()
    print("ДНО ЛЕСТНИЦЫ по формам (бит/вес, ниже которых время не падает; c = int8-цепочка):")
    print("| форма | M=32 | M=64 |")
    print("|---|--:|--:|")
    for name, K, N, m in shapes:
        print(f"| {name} | {floor_bits(name, 32, 1.0):.2f} | {floor_bits(name, 64, 1.0):.2f} |")


def _print_rows(rows, shapes, layers, nw, t16_32, t16_64):
    print("| разложение | бит/вес | по слову | ГБ | мс M=32 | x | мс M=64 | x | инстр/эл |"
          " дно, бит | flip_tf | 95 % |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda x: (x.get("flip_tf", 0))):
        spec = spec_of(r)
        bw = sum(mix_bits(r, spec, name, K, N)[0] * m * K * N for name, K, N, m in shapes) / nw
        bwp = sum(mix_bits(r, spec, name, K, N, pack="word")[0] * m * K * N
                  for name, K, N, m in shapes) / nw
        gb = bw * nw * layers / 8 / 1e9
        c = SASS_PER_ELEM[sass_key(spec)]
        t32 = mix_layer_ms(r, spec, shapes, 32, c)
        t64 = mix_layer_ms(r, spec, shapes, 64, c)
        fl = floor_bits("gate,up", 32, c)
        print(f"| {r['tag']} | {bw:.3f} | {bwp:.3f} | {gb:.1f} | {t32:.4f} | {t16_32/t32:.2f} |"
              f" {t64:.4f} | {t16_64/t64:.2f} | {c:.2f} | {fl:.1f} |"
              f" {100*r['flip_tf']:.2f} | [{100*r['flip_lo']:.2f};{100*r['flip_hi']:.2f}] |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "ref", "run", "planes", "table"])
    ap.add_argument("--model", default="gemma", choices=list(MODELS))
    ap.add_argument("--set", default="main")
    ap.add_argument("--only", default="")
    ap.add_argument("--ref", default="/mnt/d1/alex/reports/raw/EV_bits/ref_gemma.json")
    ap.add_argument("--out", default="/mnt/d1/alex/reports/raw/EV_bits/rows_gemma.json")
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--plen", type=int, default=256)
    ap.add_argument("--gen", type=int, default=192)
    ap.add_argument("--cal", type=int, default=16)
    ap.add_argument("--callen", type=int, default=1024)
    ap.add_argument("--skip", type=int, default=60000)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    {"selftest": cmd_selftest, "ref": cmd_ref, "run": cmd_run, "planes": cmd_planes,
     "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()

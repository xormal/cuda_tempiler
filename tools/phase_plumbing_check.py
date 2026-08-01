# -*- coding: utf-8 -*-
"""ПРОВЕРКА ОБВЯЗКИ фальсификаторов -- БЕЗ СЕКУНДОМЕРА (можно на занятой карте).

Отвечает ровно на один вопрос: коды nosc из phase_selfcheck_fwd_ws.py действительно
диспетчеризуются в РАЗНЫЕ ядра и каждое из них действительно ЧТО-ТО СНИМАЕТ. Признак снятия --
ответ, ОТЛИЧНЫЙ от базы (у фальсификатора он обязан быть неверным). Фальсификатор, чей ответ
СОВПАЛ с базой, ничего не снял, и его "доля" будет нулём по построению, а не по замеру.
"""

import importlib.util, math, os, sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _TOOLS]
sys.path.append("../VLLM_fa2/solutions/fa2_sm70_cutlass_grade")
import torch, fa2_sm70

D, BASE = 256, 8
PHASES = [
    ("gemm1_QK", 1508),
    ("gemm2_PV", 1308),
    ("softmax", 1108),
    ("feed", 1408),
    ("rendez", 1208),
]


def qpt(x):
    a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    return torch.clamp(torch.round(x.float() / a), -127, 127).to(
        torch.int8
    ).contiguous(), a.squeeze(-1).contiguous().float()


def bias(q):
    return (q.to(torch.int16) + 128).to(torch.uint8).view(torch.int8).contiguous()


def main():
    torch.manual_seed(0)
    pext = fa2_sm70._ext.prefill_ext()
    Sq, H, Hkv, Sk = 128, 4, 2, 1024
    sc = 1.0 / math.sqrt(D)
    q = (
        torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3
    ).contiguous()
    kf = (
        torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4
    ).contiguous()
    vf = (
        torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4
    ).contiguous()
    kq, ks = qpt(kf)
    vq, vs = qpt(vf)
    kqb, vqb = bias(kq), bias(vq)
    ksv = torch.stack([ks, vs], dim=-1).contiguous()
    call = lambda c: pext.attn_fwd_volta_i8(q, kqb, ksv, vqb, ksv, sc, True, c)
    base = call(BASE)[0].float()
    print(
        "база nosc=%d: ok, |O|=%.4f, конечно=%s"
        % (BASE, base.norm().item(), bool(torch.isfinite(base).all()))
    )
    bad = []
    for name, code in PHASES:
        try:
            o = call(code)[0].float()
        except Exception as e:
            bad.append("%s (nosc=%d): ВЫЗОВ УПАЛ -- %s" % (name, code, e))
            print(bad[-1])
            continue
        same = torch.equal(o, base)
        d = (o - base).norm().item() / max(base.norm().item(), 1e-9)
        print(
            "%-10s nosc=%-5d relL2 к базе = %.3e  %s"
            % (
                name,
                code,
                d,
                "*** СОВПАЛ С БАЗОЙ -- НИЧЕГО НЕ СНЯЛ ***"
                if same
                else "отличается (снял)",
            )
        )
        if same:
            bad.append("%s (nosc=%d) не отличается от базы" % (name, code))
    print("\n--- НЕ РАЗОБРАНО ---")
    print("  * это НЕ замер: обвязка проверена, доли фаз НЕ измерены.")
    print(
        "  * различие ответа доказывает, что ядро ДРУГОЕ, но НЕ доказывает, что снята именно"
    )
    print(
        "    названная фаза -- соответствие имени и DIAG проверяется чтением исходника."
    )
    for b in bad:
        print("  * " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

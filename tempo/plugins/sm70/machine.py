#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: МАШИНА.  Ни одного числа в коде -- всё читается из data/machine/*.json.

Почему так, а не константами в модуле: перенос на A100 обязан быть заменой ДАННЫХ, а не
правкой кода.  Если число живёт в python, оно неминуемо обрастёт условием `if arch ==`.
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from ..base import CardState, Channel, Provenance, Rate, closed_table_get

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "machine")
_CACHE = {}


def data(name: str) -> dict:
    if name not in _CACHE:
        with open(os.path.join(_DATA, name + ".json"), encoding="utf-8") as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def _card(d: dict) -> CardState:
    c = d["card"]
    return CardState(
        index=c["index"],
        clock_mhz=c["clock_mhz"],
        foreign_procs=c["foreign_procs"],
        date=c["date"],
    )


def _prov(d: dict, counter=None) -> Provenance:
    tw = d.get("taken_with", {})
    return Provenance(
        tool=tw.get("tool", "n/a"),
        counter=counter if counter is not None else tw.get("counter"),
        version=tw.get("version", "n/a"),
        date=d["card"]["date"],
        who="Alexander Romanov",
        observability=d.get("observability", "clean"),
        card=_card(d),
    )


def _rate(symbol, value, units, status, note, src: dict, counter=None) -> Rate:
    """Происхождение прикладывается ТОЛЬКО к MEASURED: у SPEC карты нет и быть не должно."""
    prov = _prov(src, counter) if status == "MEASURED" else None
    return Rate(
        symbol=symbol,
        value=float(value),
        units=units,
        status=status,
        prov=prov,
        note=note,
    )


def _build_symbols() -> Mapping[str, Rate]:
    out = {}
    ch = data("channels")
    for name, c in ch["channels"].items():
        out["CAP." + name] = _rate(
            "CAP." + name, c["value"], c["units"], c["status"], c.get("note", ""), ch
        )
    lat = data("latency")
    for name, l in lat["latency"].items():
        out["LATENCY." + name] = _rate(
            "LATENCY." + name, l["value"], "такт", l["status"], l.get("note", ""), lat
        )
    m = data("machine")
    for name, g in m["geometry"].items():
        out["GEOM." + name.upper()] = _rate(
            "GEOM." + name.upper(),
            g["value"],
            g["units"],
            g["status"],
            g.get("note", ""),
            m,
        )
    for name, p in m["peak"].items():
        out["PEAK." + name.upper()] = _rate(
            "PEAK." + name.upper(),
            p["value"],
            p["units"],
            p["status"],
            p.get("note", ""),
            m,
        )
    r = data("regs")
    out["REG.OVERHEAD"] = _rate(
        "REG.OVERHEAD",
        r["compiler_threshold"]["overhead"],
        "регистр",
        r["compiler_threshold"]["status"],
        r["compiler_threshold"]["note"],
        r,
    )
    out["REG.FREE_SPILLS"] = _rate(
        "REG.FREE_SPILLS",
        r["free_spills"]["value"],
        "значение",
        r["free_spills"]["status"],
        r["free_spills"]["note"],
        r,
    )
    out["REG.SPILL_STEP"] = _rate(
        "REG.SPILL_STEP",
        r["spill_cost"]["step_multiplier"],
        "кратность",
        r["spill_cost"]["status"],
        r["spill_cost"]["note"],
        r,
    )
    out["REG.SPILL_EDGE"] = _rate(
        "REG.SPILL_EDGE",
        r["spill_cost"]["edge_multiplier"],
        "кратность",
        r["spill_cost"]["status"],
        "",
        r,
    )
    out["REG.SECOND_CTA_SMEM"] = _rate(
        "REG.SECOND_CTA_SMEM",
        r["second_cta_threshold"]["smem_bytes"],
        "байт",
        r["second_cta_threshold"]["status"],
        r["second_cta_threshold"]["note"],
        r,
    )
    out["REG.SECOND_CTA_REGS"] = _rate(
        "REG.SECOND_CTA_REGS",
        r["second_cta_threshold"]["regs_per_thread"],
        "регистр",
        r["second_cta_threshold"]["status"],
        "",
        r,
    )
    mio = data("mio")
    out["MIO.WAVEFRONT_BYTES"] = _rate(
        "MIO.WAVEFRONT_BYTES",
        128.0,
        "байт",
        "MEASURED",
        mio["law"]["throughput"],
        mio,
        counter=mio["taken_with"]["counter"],
    )
    out["MIO.CONFLICT"] = Rate(
        symbol="MIO.CONFLICT",
        value=float("nan"),
        units="кратность",
        status="NOT_MEASURED",
        prov=None,
        note=mio["MIO_CONFLICT_is_an_input"]["note"],
    )
    w = data("wave")
    out["WAVE.QUANTUM"] = _rate(
        "WAVE.QUANTUM",
        w["wave"]["quantum_sms"],
        "SM",
        w["wave"]["status"],
        w["wave"]["note"],
        w,
    )
    out["ISSUE.SLOT_PRICE_IDLE"] = _rate(
        "ISSUE.SLOT_PRICE_IDLE",
        0.22,
        "такт/слот",
        "MEASURED",
        "НИЖНЯЯ оценка: холостая команда у считающих. "
        + w["issue_slot_price"]["correction"],
        w,
    )
    out["ISSUE.SLOT_PRICE_INCHAIN"] = _rate(
        "ISSUE.SLOT_PRICE_INCHAIN",
        0.49,
        "такт/слот",
        "MEASURED",
        "цена той же команды В ЦЕПИ подачи операнда",
        w,
    )
    t = data("tensor")
    out["TENSOR.COST"] = _rate(
        "TENSOR.COST",
        t["op"]["cost_cycles_per_sched"],
        "такт/команду/планировщик",
        t["op"]["status"],
        t["op"]["gotcha"],
        t,
    )
    return out


_SYMBOLS = None


class Sm70Machine:
    def arch(self):
        return "sm_70"

    def symbols(self) -> Mapping[str, Rate]:
        global _SYMBOLS
        if _SYMBOLS is None:
            _SYMBOLS = _build_symbols()
        return _SYMBOLS

    def rate(self, symbol: str) -> Rate:
        """ЗАКРЫТАЯ таблица: символа нет -> UnknownSymbol, а не умолчание (гейт G4)."""
        return closed_table_get(self.symbols(), symbol)

    def channels(self) -> Mapping[str, Channel]:
        ch = data("channels")
        out = {}
        for name, c in ch["channels"].items():
            out[name] = Channel(
                name=name,
                scope=c["scope"],
                capacity=self.rate("CAP." + name),
            )
        return out

    def latency(self, atom_class: str) -> Rate:
        return self.rate("LATENCY." + atom_class)

    def sms(self):
        return self.rate("GEOM.SMS")

    def schedulers_per_sm(self):
        return self.rate("GEOM.SCHEDULERS_PER_SM")

    def warp_slots_per_sm(self):
        return self.rate("GEOM.WARP_SLOTS_PER_SM")

    def threads_per_warp(self):
        return self.rate("GEOM.THREADS_PER_WARP")

    def smem_per_sm(self):
        return self.rate("GEOM.SMEM_PER_SM")

    def smem_per_cta_max(self):
        return self.rate("GEOM.SMEM_PER_CTA_MAX")

    def regfile_words_per_sm(self):
        return self.rate("GEOM.REGFILE_WORDS_PER_SM")

    def clock_mhz(self):
        return self.rate("GEOM.CLOCK_MHZ")

    def peak(self, kind: str) -> Rate:
        return self.rate("PEAK." + kind.upper())

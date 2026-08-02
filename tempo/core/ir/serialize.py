#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Сериализация списка атомов: схема `tempo/atoms/1`.

Нужна ровно для двух вещей: диффа между двумя вариантами и УСТОЙЧИВОГО ХЭША, которым
именуется каталог поставки.  Хэш обязан быть устойчив к порядку словарей, иначе одна и та же
гиперформа получит два имени и поставка разъедется с манифестом.
"""

from __future__ import annotations

import hashlib
import json

from tempo.plugins.base import Atom, AtomKind, Dep

SCHEMA = "tempo/atoms/1"


def to_json(atoms, meta=None) -> str:
    doc = {
        "schema": SCHEMA,
        "meta": meta or {},
        "atoms": [
            {
                "uid": a.uid,
                "kind": a.kind.value if isinstance(a.kind, AtomKind) else str(a.kind),
                "op_id": a.op_id,
                "footprint": {k: round(float(v), 6) for k, v in sorted(a.footprint.items())},
                "latency": round(float(a.latency), 6),
                "token": a.token,
                "region": a.region,
                "deps": [
                    {"src": d.src_uid, "type": d.type, "distance": d.distance,
                     "min_gap": round(float(d.min_gap), 6)}
                    for d in a.deps
                ],
            }
            for a in atoms
        ],
    }
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2)


def from_json(text: str):
    doc = json.loads(text)
    if doc.get("schema") != SCHEMA:
        raise ValueError("чужая схема %r; ожидалась %r" % (doc.get("schema"), SCHEMA))
    out = []
    for a in doc["atoms"]:
        out.append(
            Atom(
                uid=a["uid"],
                kind=AtomKind(a["kind"]),
                op_id=a["op_id"],
                footprint=dict(a["footprint"]),
                latency=a["latency"],
                token=a.get("token"),
                deps=tuple(
                    Dep(d["src"], d["type"], d.get("distance", 0), d.get("min_gap", 0.0))
                    for d in a.get("deps", ())
                ),
                region=a.get("region", "body"),
            )
        )
    return out


def stable_hash(atoms, meta=None) -> str:
    return hashlib.sha256(to_json(atoms, meta).encode("utf-8")).hexdigest()[:16]


def diff(a, b) -> list:
    """Различия по каналам -- то, что читает человек, когда два варианта «одинаковы»."""
    from .atom import channel_load

    la, lb = channel_load(a), channel_load(b)
    keys = sorted(set(la) | set(lb))
    return [(k, la.get(k, 0.0), lb.get(k, 0.0)) for k in keys if abs(la.get(k, 0.0) - lb.get(k, 0.0)) > 1e-9]

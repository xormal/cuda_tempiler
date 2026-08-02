# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
#
# Пять целей. Первые три НЕ ТРЕБУЮТ КАРТЫ и обязаны быть зелёными всегда.
PY ?= python3

.PHONY: help doctor gates selftest laws first-number ship clean

help:
	@echo "make doctor       -- окружение и плагины (карту не трогает)"
	@echo "make gates        -- гейты непротекания G1..G11 (карта не нужна)"
	@echo "make selftest     -- самопроверки плагинов, ПРИЁМКА ОТСЕКАТЕЛЯ и инструменты"
	@echo "make laws         -- РЕЕСТР ЗАКОНОВ: опись, проверки и сверка docs/LAWS.md"
	@echo "make first-number -- ПЕРВОЕ ЧИСЛО: наивный вход -> отношение к сопернику (НУЖНА СВОБОДНАЯ КАРТА)"
	@echo "make ship         -- проверить поставку ядра (карта не нужна)"
	@echo "make anchors      -- ЯКОРЯ, требующие СВОБОДНОЙ КАРТЫ (прибор + стенд периода)"

doctor:
	$(PY) -m tempo.cli doctor

gates:
	$(PY) tests/test_gates.py

selftest:
	$(PY) -m tempo.cli selftest
	$(PY) tests/test_pipeline.py
	$(PY) tests/test_prune_acceptance.py
	$(PY) tests/test_tools_selftest.py

# Якоря, которым НУЖНА СВОБОДНАЯ КАРТА (и приборный -- ещё и права на профилировщик).
# Проверять занятость карт ДО запуска: замер с соседом недействителен.
anchors:
	$(PY) tests/test_tools_selftest.py --with-card

# ВНИМАНИЕ: цель мерит. Дисциплина обязательна: свободная карта, закреплённые частоты,
# парные отношения с чередованием ВНУТРИ раунда, медиана ОТНОШЕНИЙ, счёт чужих процессов
# ДО и ПОСЛЕ. Замер с соседом на карте НЕДЕЙСТВИТЕЛЕН.
first-number:
	$(PY) bench/tempolate_gemm.py --help

# РЕЕСТР ЗАКОНОВ.  Три шага: опись -> проверки записи/места/привязки -> сверка чисел якорей
# с ТЕЛАМИ отчётов (последнее пропускается с причиной, если каталога отчётов нет).
laws:
	$(PY) -m tempo.cli laws
	$(PY) -m tempo.cli laws --check
	$(PY) tools/laws_from_reports.py anchors

ship:
	$(PY) kernels/shipped/gemm_fp16/sm_70/test_gate.py

clean:
	rm -rf build/tempo
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

#!/bin/bash
# ФАЛЬСИФИКАТОР ВОРОТ ДРЕЙФА. Ворота, которые никогда не падали, не доказаны.
# Работает в ПЕСОЧНИЦЕ (копия боевых файлов), боевое дерево не трогается ничем.
# Проверяется ТРИ отказа + один успех:
#   A. исходник изменился (md5)                       -> должен упасть с кодом 2
#   B. якорь пропал (md5 в наложении подогнан)        -> должен упасть с кодом 2
#   C. якорь встретился ДВАЖДЫ (блок продублирован)   -> должен упасть с кодом 2
#   D. нетронутая песочница                           -> должен пройти с кодом 0
set -u
PROD=../VLLM_fa2/solutions/fa2_sm70_cutlass_grade
S=./profiled/bwd/drift
REL=fa2_src/fmha_kernel/kernel_backward.h
rm -rf $S; mkdir -p $S/root/$(dirname $REL) $S/out
cp $PROD/$REL $S/root/$REL
cp $PROD/fa2_src/fmha_kernel/fused_qk_gradk.h $S/root/fa2_src/fmha_kernel/
cp ./profiled/bwd/overlay_bwd.json $S/ov.json

run() {  # $1 = имя случая
  PHASE_PROD_ROOT=$S/root PHASE_TWIN_OUT=$S/out PHASE_OVERLAY=$S/ov.json \
    python3 ./tools/gen_bwd_twin.py > $S/$1.out 2> $S/$1.err
  echo $?
}

expect() { # $1=имя $2=ожидаемый код
  C=$(run $1)
  if [ "$C" = "$2" ]; then echo "  [ОК] $1: код $C (ожидался $2)";
  else echo "  [ПРОВАЛ ВОРОТ] $1: код $C, ожидался $2"; fi
  sed -n '1,4p' $S/$1.err | sed 's/^/      /'
}

echo "D. нетронутая песочница (ворота обязаны ПРОПУСТИТЬ):"
expect D 0

echo "A. в боевой файл дописан один пробел (md5 поехал):"
printf ' \n' >> $S/root/$REL
expect A 2

echo "B. md5 в наложении подогнан под изменённый файл, но якорь испорчен:"
python3 - "$S" "$REL" <<'PY'
import hashlib, json, sys, re
S, REL = sys.argv[1], sys.argv[2]
p = S + "/root/" + REL
t = open(p, encoding="utf-8").read()
# портим ровно одну строку, которая входит в якорь правки (тело dS)
t = t.replace("accum = accum * scale;", "accum = accum * scale;  // ДРЕЙФ", 1)
open(p, "w", encoding="utf-8").write(t)
ov = json.load(open(S + "/ov.json", encoding="utf-8"))
ov[0]["md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()   # "кто-то обновил md5"
json.dump(ov, open(S + "/ov.json", "w", encoding="utf-8"), ensure_ascii=False)
PY
expect B 2

echo "C. якорь встречается ДВАЖДЫ (блок продублирован, md5 снова подогнан):"
cp $PROD/$REL $S/root/$REL
python3 - "$S" "$REL" <<'PY'
import hashlib, json, sys
S, REL = sys.argv[1], sys.argv[2]
p = S + "/root/" + REL
t = open(p, encoding="utf-8").read()
ov = json.load(open(S + "/ov.json", encoding="utf-8"))
a = ov[0]["edits"][0]["anchor"]
assert t.count(a) == 1
t = t.replace(a, a + "\n#if 0\n" + a + "\n#endif\n", 1)     # второй экземпляр якоря
open(p, "w", encoding="utf-8").write(t)
ov[0]["md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()
json.dump(ov, open(S + "/ov.json", "w", encoding="utf-8"), ensure_ascii=False)
PY
expect C 2

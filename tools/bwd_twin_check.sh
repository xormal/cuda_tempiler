#!/bin/bash
# НУЛЕВАЯ ЦЕНА ДВОЙНИКА: при маске 0 (и без -D пломбы) двойник обязан давать ТО ЖЕ ЯДРО, что и
# боевое дерево. Сравнение идёт по БАЙТАМ секции .text ядра, а не по числу регистров: одинаковые
# 255 регистров бывают и у разного кода.
# ПОЧЕМУ НЕ ВЕСЬ .cubin: в него зашит хеш препроцессированного исходника, а комментарии разметки в
# заголовке его меняют, не меняя ни одной команды. Поэтому сверяются: секции ELF (имена/типы/
# размеры), БАЙТЫ .text ядра, регистры/кадр/разлив.
set -e
cd ./profiled/bwd
TILE=${1:-128,64,128,true}
D=out/zerocost; mkdir -p $D
# ОДИН И ТОТ ЖЕ путь исходника у обеих сборок -- иначе разойдутся строки таблицы имён.
./build.sh $D/k BASE "$TILE" > $D/base.res 2>&1; cp $D/k.cubin $D/k_BASE.cubin; cp $D/k.sass $D/k_BASE.sass
./build.sh $D/k TWIN "$TILE" > $D/twin.res 2>&1; cp $D/k.cubin $D/k_TWIN.cubin; cp $D/k.sass $D/k_TWIN.sass
echo "боевое : $(tr '\n' ' ' < $D/base.res)"
echo "двойник: $(tr '\n' ' ' < $D/twin.res)"
for f in k_BASE k_TWIN; do
  L=$(readelf -S -W $D/$f.cubin 2>/dev/null | grep '\.text\.' | head -1)
  O=$((16#$(echo $L | awk '{print $5}'))); S=$((16#$(echo $L | awk '{print $6}')))
  dd if=$D/$f.cubin of=$D/$f.textbin bs=1 skip=$O count=$S status=none
  readelf -S -W $D/$f.cubin 2>/dev/null | grep '^ *\[' | awk '{print $3,$4,$7}' > $D/$f.sec
done
diff -q $D/k_BASE.sec $D/k_TWIN.sec > /dev/null && echo "[ОК] секции ELF совпадают" || echo "[ПРОВАЛ] секции ELF разошлись"
cmp $D/k_BASE.textbin $D/k_TWIN.textbin && echo "[ОК] БАЙТЫ .text ЯДРА ПОБАЙТОВО ОДИНАКОВЫ ($(stat -c%s $D/k_BASE.textbin) Б)" \
  || echo "[ПРОВАЛ] .text разошёлся -- двойник НЕ бесплатен"
cmp $D/k_BASE.sass $D/k_TWIN.sass && echo "[ОК] разбор SASS совпадает" || echo "[ПРОВАЛ] SASS разошёлся"

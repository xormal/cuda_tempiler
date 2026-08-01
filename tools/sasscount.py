import re, sys, collections, os

INSTR = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*?);", re.M)


def counts(path):
    txt = open(path).read()
    c = collections.Counter()
    n = 0
    for m in INSTR.finditer(txt):
        ins = m.group(1).strip()
        ins = re.sub(r"^@!?P\d+\s+", "", ins)
        op = ins.split()[0]
        base = op.split(".")[0]
        c[base] += 1
        n += 1
    c["TOTAL"] = n
    return c

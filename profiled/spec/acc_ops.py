import re, sys, collections


def ops(p):
    out = []
    for ln in open(p):
        m = re.search(r"/\*[0-9a-f]{4}\*/\s+(.*?);", ln)
        if m:
            out.append(re.sub(r"\s+", " ", m.group(1).strip()))
    return out


def opc(o):
    c = collections.Counter()
    for i in o:
        t = i.split()
        k = t[1] if t and t[0].startswith("@") else (t[0] if t else "")
        c[k.split(".")[0]] += 1
    return c


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    A, B = ops(a), ops(b)
    print(f"команд: {len(A)} против {len(B)}")
    print("ПОКОМАНДНО СОВПАДАЕТ" if A == B else "РАЗОШЁЛСЯ")
    if A != B:
        for i, (x, y) in enumerate(zip(A, B)):
            if x != y:
                print(f"  первая разница @{i}: {x!r} vs {y!r}")
                break

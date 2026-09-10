import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt_v2 as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)

def classify(cot):
    # scenario type (A1), matching AXIS_DISTRIBUTION_800.md definitions
    if 'it is a QUERY-only operator that we resolve by default' in cot or 'appears only in the QUERY, so no example fixes it' in cot:
        return 'query-only'
    if 'not arithmetically solvable' in cot or 'neither reading can solve' in cot:
        return 'not-arith-solvable'
    tail = cot.split('Now solve the QUERY')[-1] if 'Now solve the QUERY' in cot else cot[-400:]
    if 'a∥b' in tail or 'b∥a' in tail:
        return 'concat'
    return 'deduce'

def reading(cot):
    m = re.search(r'reading order = (rightward|leftward)', cot)
    if m: return m.group(1)
    if 'not arithmetically solvable' in cot or 'neither reading' in cot: return 'unsolved'
    if 'QUERY-only' in cot or 'a∥b; answer' in cot: return 'short-circuit'
    return 'other'

dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
rows=[]; to=[]
t0=time.time()
for i,d in enumerate(dirs,1):
    name=os.path.basename(d)
    try: signal.alarm(180); cot=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(name); continue
    except Exception as e: signal.alarm(0); print("ERR",name,str(e)[:40],flush=True); continue
    L=tok(cot)
    rows.append((name, L, L<7680, classify(cot), reading(cot)))
    if i%150==0: print(f"  ...{i} ({time.time()-t0:.0f}s)",flush=True)

import collections
def table(title, rows, keyidx):
    print(f"\n## {title}")
    full=collections.Counter(r[keyidx] for r in rows)
    under=collections.Counter(r[keyidx] for r in rows if r[2])
    over=collections.Counter(r[keyidx] for r in rows if not r[2])
    keys=sorted(full, key=lambda k:-full[k])
    print(f"{'value':<22}{'full':>8}{'under7680':>11}{'over(lost)':>12}{'keep%':>8}")
    for k in keys:
        kp = 100*under[k]/full[k] if full[k] else 0
        print(f"{k:<22}{full[k]:>8}{under[k]:>11}{over[k]:>12}{kp:>7.0f}%")
    print(f"{'TOTAL':<22}{len(rows):>8}{sum(r[2] for r in rows):>11}{sum(not r[2] for r in rows):>12}")

print(f"\n=== regenerated {len(rows)} (timeouts {len(to)}: {to[:6]}) ===")
table("A1 scenario type", rows, 3)
table("A2 reading", rows, 4)
# write per-puzzle csv
with open('_dist_435.csv','w') as f:
    f.write('name,tokens,under_cap,scenario,reading\n')
    for r in rows: f.write(f"{r[0]},{r[1]},{int(r[2])},{r[3]},{r[4]}\n")
print("\nwrote _dist_435.csv")

import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver')
import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def extract(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
CAP=7680
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'question.txt')) and os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
oldL=[]; newL=[]; boxchg=[]; gtO=0; gtN=0; capO=0; capN=0; to=[]; n=0
trO=0; trN=0; t0=time.time()
emptybox=[]; forbidden=[]; ann_bad=[]; ann_ok=0; hdr_ok=0; binding=0
trN_lens=[]
for d in dirs:
    cp=os.path.join(d,'track','tree_cot.txt')
    old=open(cp).read(); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try:
        signal.alarm(150); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:80],flush=True); continue
    n+=1
    lo,ln=tok(old),tok(new); bo,bn=extract(old),extract(new)
    oldL.append(lo); newL.append(ln)
    if bo!=bn: boxchg.append((os.path.basename(d),bo,bn))
    gtO+=(bo==gold); gtN+=(bn==gold); capO+=(lo<CAP); capN+=(ln<CAP)
    trO += (bo==gold and lo<CAP); is_tr=(bn==gold and ln<CAP); trN += is_tr
    if is_tr: trN_lens.append(ln)
    # sweep
    if bn is None or bn=='': emptybox.append(os.path.basename(d))
    if ('= unknown' in new) or ('exotic' in new) or re.search(r'\bguess\b', new): forbidden.append(os.path.basename(d))
    unsolv = 'Equations are arithmetically unsolvable' in new
    ann = bool(re.search(r"is left, [A-Z] is not solvable", new))
    if ann and not unsolv: ann_bad.append(os.path.basename(d))   # annotation must ONLY appear in an unsolvable CoT
    if ann and unsolv: ann_ok+=1
    if "narrow each symbol's domain by comparing" in new: hdr_ok+=1
    binding += new.count('binary search')
    open(cp,'w').write(new)
    if n%200==0: print(f"  ...{n} done ({time.time()-t0:.0f}s)",flush=True)
md=lambda x:sorted(x)[len(x)//2] if x else 0
print("\n================ REGEN — item2/3/header EFFECT ================")
print(f"regenerated {n} (timeouts {len(to)}: {to})")
print(f"CoT length (tokens): OLD median {md(oldL)} total {sum(oldL):,}  ->  NEW median {md(newL)} total {sum(newL):,}  ({sum(newL)-sum(oldL):+,})")
print(f"under {CAP} cap:   OLD {capO}/{n}  ->  NEW {capN}/{n}  ({capN-capO:+d})")
print(f"GT-match:         OLD {gtO}/{n} ({100*gtO/n:.1f}%)  ->  NEW {gtN}/{n} ({100*gtN/n:.1f}%)")
print(f"TRAINABLE (GT-True & under-cap): OLD {trO}  ->  NEW {trN}  ({trN-trO:+d})")
print(f"TRAINABLE length: median {md(trN_lens)} max {max(trN_lens) if trN_lens else 0}")
print(f"boxed-answer changed: {len(boxchg)}")
for nm,a,b in boxchg[:12]: print(f"   {nm}: {a!r} -> {b!r}")
print("\n--- sweep ---")
print(f"empty boxes: {len(emptybox)} {emptybox[:10]}")
print(f"forbidden wording (guess/unknown/exotic): {len(forbidden)} {forbidden[:10]}")
print(f"'cannot be solved' annotation in NON-unsolvable CoT (must be 0): {len(ann_bad)} {ann_bad[:10]}")
print(f"'cannot be solved' annotation in unsolvable CoT (correct): {ann_ok}")
print(f"new §4 header present: {hdr_ok}")
print("============================================================")

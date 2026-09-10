import sys, re, os, glob, signal, time, importlib.util
sys.path.insert(0,'_cryptarithm_solver')
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    if not t: return None
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
def load(name):
    spec=importlib.util.spec_from_file_location(name, f'_cryptarithm_solver/{name}.py')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
CAP=7680
dirs=[x.rstrip('/') for x in sorted(glob.glob('*/')) if os.path.exists(os.path.join(x,'track','tree_cot.txt'))]
def measure(gc, label):
    tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
    uc_gt=uc=gt=0; n=0; t0=time.time()
    for d in dirs:
        gold=open(os.path.join(d,'answer.txt')).read().strip()
        try: signal.alarm(150); c=gc.gen_cot(d); signal.alarm(0)
        except Exception: signal.alarm(0); continue
        n+=1; L=tok(c); b=box(c); g=(b==gold)
        gt+=g; uc+=(L<CAP); uc_gt+=(L<CAP and g)
    print(f"{label:<34} n={n} | GT-True {gt} | under-cap {uc} | TRAINABLE(GT-True & under-cap) {uc_gt}",flush=True)
print("measuring pre-v2 (simple format, session start)...")
measure(load('_gen_crypt_pre_v2_260613'), 'pre-v2 simple (session start)')
print("measuring current (full verbose) ...")
measure(load('_gen_crypt'), 'current full-verbose')

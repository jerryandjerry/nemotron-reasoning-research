#!/usr/bin/env python3
"""Rigorous verification of the three proposed all-different GAC rules for thread gac-alldiff.

We monkeypatch Engine.propagate: run the ORIGINAL propagate to fixpoint first, then apply a
candidate rule and count ONLY new prunings (changes the original fixpoint did NOT already make).

Three measurement modes:
  --count   : instrument propagate, count NEW fires (per rule), Hall-set/naked-subset size distribution.
  --ab RULE : full A/B over 800 puzzles -> boxed-answer changes, token delta, over->under rescues.
  --sound   : structural soundness: capture each puzzle's winning solution A; at EVERY node the rule
              would fire, verify it never prunes a gold value from a symbol's domain.

Rules:
  hall     = naked_subset_k for k=2..5 deficiency (contradiction) + Hall cut (exactly-k confine).
  naked    = naked_subset_k k=2,3 only (the headline 'naked pair/triple' rule), gated to fire only when
             a victim becomes a singleton or empties (a branch actually disappears).
  matching = Regin feasibility: no perfect symbol->digit matching => DEAD node (contradiction only).
"""
import copy, glob, os, sys, json
from itertools import combinations
import cot_generator as cg
import _gen_crypt as gc

PREFIXES = ('arithmetic','little_endian','mixed_concat','pure_concat','query_unseen_concat','mixed_concat_little_endian')

def all_names():
    out=[]
    for d in sorted(glob.glob('*/')):
        name=d.rstrip('/').split('/')[-1]
        if os.path.exists(os.path.join(d,'question.txt')) and any(name.startswith(p) for p in PREFIXES):
            out.append(name)
    return out

# ---------------- matching feasibility (Kuhn) ----------------
def feasible(syms, dom):
    matchR={}
    def kuhn(s, seen, adj):
        for v in adj[s]:
            if v in seen: continue
            seen.add(v)
            if v not in matchR or kuhn(matchR[v], seen, adj):
                matchR[v]=s; return True
        return False
    adj={s:sorted(dom[s]) for s in syms}
    m=0
    for s in syms:
        if kuhn(s, set(), adj): m+=1
    return m==len(syms)

def smallest_deficient(syms, dom, maxk=6):
    """Smallest S with |union(domains)|<|S|. By Hall, exists iff no perfect matching."""
    ss=sorted(syms, key=lambda s: len(dom[s]))
    for k in range(2, maxk+1):
        pool=[s for s in ss if len(dom[s])<k]
        for combo in combinations(pool, k):
            union=set().union(*(dom[s] for s in combo))
            if len(union)<k: return k, combo, union
    return None

# ---------------- rule implementations (operate on dom in place, return action + info) ----------------
def apply_hall(syms, lm, dom, emit, kmax=5, naked_only=False, gate_singleton=False):
    """One Hall/naked-subset action per call (smallest k). naked_only: skip the deficiency-contradiction
    half? No -- both halves are part of naked-subset. gate_singleton: only emit a 'cut' if a victim becomes
    singleton/empty. Returns ('contra'|'cut'|None, k, info)."""
    small=[s for s in syms if 2<=len(dom[s])<=kmax]
    for k in range(2, kmax+1):
        pool=[s for s in small if len(dom[s])<=k]
        for combo in combinations(pool, k):
            union=set().union(*(dom[s] for s in combo))
            if len(union)<k:
                if emit: emit(f"All-different (Hall): {', '.join(lm[s] for s in combo)} can together only be {{{','.join(map(str,sorted(union)))}}} — {k} symbols but {len(union)} digits, impossible — no.")
                return 'contra', k, (combo, union)
            if len(union)==k:
                victims=[t for t in syms if t not in combo and (dom[t]&union)]
                if not victims: continue
                makes_singleton=any(len(dom[t]-union) in (0,1) for t in victims)
                if gate_singleton and not makes_singleton:
                    continue
                emptied=None
                for t in victims:
                    dom[t]=dom[t]-union
                    if not dom[t]: emptied=t
                if emptied is not None:
                    if emit: emit(f"All-different (Hall): {lm[emptied]} loses all of {{{','.join(map(str,sorted(union)))}}} — no.")
                    return 'contra', k, (combo, union)
                if emit: emit(f"All-different (Hall): {', '.join(lm[s] for s in combo)} use up {{{','.join(map(str,sorted(union)))}}}, remove from {', '.join(lm[t] for t in victims)}.")
                return 'cut', k, (combo, union)
    return None, None, None

# ---------------- COUNT harness ----------------
def make_patched(rule, kmax=5, gate_singleton=False, counter=None):
    ORIG=gc.Engine.propagate
    def patched(self, dom, locked, emit):
        while True:
            ok=ORIG(self, dom, locked, emit)
            if not ok: return False
            # snapshot to detect NEW pruning only
            if rule=='matching':
                if not feasible(self.syms, dom):
                    w=smallest_deficient(self.syms, dom)
                    if counter is not None:
                        counter['fire']+=1; counter['contra']+=1
                        if w: counter['ksize'][w[0]]=counter['ksize'].get(w[0],0)+1
                        else: counter['nowitness']+=1
                    return False
                return True
            else:  # hall or naked
                k_lo = 2 if rule in ('hall','naked') else 2
                k_hi = (3 if rule=='naked' else kmax)
                act,k,info=apply_hall(self.syms, self.lm, dom, emit, kmax=k_hi,
                                      gate_singleton=(rule=='naked' and gate_singleton))
                if act is None: return True
                if counter is not None:
                    counter['fire']+=1; counter[act]+=1
                    counter['ksize'][k]=counter['ksize'].get(k,0)+1
                if act=='contra': return False
                # 'cut' changed dom -> loop original fixpoint again
    return patched

def run_count(names, rule, kmax=5, gate_singleton=False):
    counter={'fire':0,'contra':0,'cut':0,'ksize':{},'nowitness':0}
    ORIG=gc.Engine.propagate
    gc.Engine.propagate=make_patched(rule, kmax=kmax, gate_singleton=gate_singleton, counter=counter)
    fired_puzzles=set()
    per_before=dict(counter)
    for name in names:
        f0=counter['fire']
        try: gc.gen_cot(name)
        except Exception: pass
        if counter['fire']>f0: fired_puzzles.add(name)
    gc.Engine.propagate=ORIG
    counter['puzzles_fired']=len(fired_puzzles)
    counter['fired_names']=sorted(fired_puzzles)
    return counter

# ---------------- A/B harness ----------------
def boxed(c):
    i=c.rfind('\\boxed{'); j=c.find('}', i); return c[i+7:j] if i>=0 else None

def run_ab(names, rule, base, kmax=5, gate_singleton=False):
    from tokenizers import Tokenizer
    TOK=Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
    ORIG=gc.Engine.propagate
    gc.Engine.propagate=make_patched(rule, kmax=kmax, gate_singleton=gate_singleton, counter=None)
    rows={}
    for name in names:
        try:
            c=gc.gen_cot(name); rows[name]={'tok':len(TOK.encode(c).ids),'ans':boxed(c)}
        except Exception as e:
            rows[name]={'err':repr(e)}
    gc.Engine.propagate=ORIG
    ans_changed=[]; tok_b=tok_a=ob=oa=0; rescued=[]; pushed_over=[]; errs=[]
    for name in names:
        b=base[name]; a=rows[name]
        if 'err' in a: errs.append((name,a['err'])); continue
        if a.get('ans')!=b.get('ans'): ans_changed.append((name,b.get('ans'),a.get('ans')))
        tb,ta=b.get('tok',0),a.get('tok',0); tok_b+=tb; tok_a+=ta
        bo,aoo=tb>8192, ta>8192; ob+=bo; oa+=aoo
        if bo and not aoo: rescued.append((name,tb,ta))
        if aoo and not bo: pushed_over.append((name,tb,ta))
    return {'rows':rows,'ans_changed':ans_changed,'tok_b':tok_b,'tok_a':tok_a,
            'over_b':ob,'over_a':oa,'rescued':rescued,'pushed_over':pushed_over,'errs':errs}

# ---------------- SOUNDNESS structural harness ----------------
def capture_solutions(names):
    """Run baseline (orig propagate) and capture the winning solution A + mode per puzzle."""
    ORIG=gc.Engine.search
    sols={}
    cur=[None]
    def wrapped(self, dom, locked, emit, depth=0):
        r=ORIG(self, dom, locked, emit, depth)
        if r is not None and cur[0] is not None and cur[0] not in sols:
            sols[cur[0]]={'A':{s:int(v) for s,v in r[0].items()},'mode':self.mode}
        return r
    gc.Engine.search=wrapped
    for name in names:
        cur[0]=name
        try: gc.gen_cot(name)
        except Exception: pass
    gc.Engine.search=ORIG
    return sols

def run_sound(names, rule, sols, kmax=5):
    """At every node the rule fires, verify it never removes a gold value from a symbol's domain.
    We capture (dom,gold) at each fire by patching propagate to test the rule on the post-fixpoint dom
    BEFORE applying it, and checking each pruned value is not a gold value of that symbol."""
    ORIG=gc.Engine.propagate
    violations=[]
    cur=[None]
    fire_nodes=[0]
    def patched(self, dom, locked, emit):
        while True:
            ok=ORIG(self, dom, locked, emit)
            if not ok: return False
            gA=sols.get(cur[0],{}).get('A')
            gold_ok = gA is not None and self.mode==sols[cur[0]]['mode'] and all(
                (gA[s] in dom[s]) for s in self.syms if s in gA)
            if rule=='matching':
                if not feasible(self.syms, dom):
                    fire_nodes[0]+=1
                    # if the gold assignment is still entirely in-domain here, infeasibility is a CONTRADICTION
                    # that wrongly kills a node containing the gold -> UNSOUND.
                    if gold_ok:
                        violations.append((cur[0],'matching killed node still containing gold'))
                    return False
                return True
            else:
                dom_before={s:set(dom[s]) for s in self.syms}
                k_hi=(3 if rule=='naked' else kmax)
                act,k,info=apply_hall(self.syms, self.lm, dom, None, kmax=k_hi,
                                      gate_singleton=(rule=='naked'))
                if act is None: return True
                fire_nodes[0]+=1
                if act=='contra':
                    if gold_ok:
                        violations.append((cur[0],f'{rule} contra killed node still containing gold (k={k})'))
                    return False
                # cut: check no gold value was removed
                if gA is not None and self.mode==sols[cur[0]]['mode']:
                    for s in self.syms:
                        if s in gA and gA[s] in dom_before[s] and gA[s] not in dom[s]:
                            violations.append((cur[0],f'{rule} cut removed gold {gA[s]} from {self.lm[s]} (k={k})'))
    gc.Engine.propagate=patched
    for name in names:
        cur[0]=name
        try: gc.gen_cot(name)
        except Exception: pass
    gc.Engine.propagate=ORIG
    return {'fire_nodes':fire_nodes[0],'violations':violations}

if __name__=='__main__':
    mode=sys.argv[1]
    sel=sys.argv[2] if len(sys.argv)>2 else 'all'
    base=json.load(open('_baseline_fresh.json'))
    if sel=='over': names=sorted(n for n,v in base.items() if v.get('tok',0)>8192)
    else: names=all_names()
    if mode=='count':
        for rule in ['matching','naked','hall']:
            c=run_count(names, rule)
            print(f"=== COUNT rule={rule} set={sel} puzzles={len(names)} ===")
            print(f"  NEW fires={c['fire']} contra={c['contra']} cut={c['cut']} puzzles_fired={c['puzzles_fired']} ksize={dict(sorted(c['ksize'].items()))} nowitness={c.get('nowitness')}")
    elif mode=='ab':
        rule=sys.argv[3] if len(sys.argv)>3 else 'hall'
        r=run_ab(names, rule, base)
        print(f"=== A/B rule={rule} set={sel} puzzles={len(names)} ===")
        print(f"  ANSWERS CHANGED: {len(r['ans_changed'])} {r['ans_changed'][:12]}")
        print(f"  tokens: base={r['tok_b']} rule={r['tok_a']} delta={r['tok_a']-r['tok_b']} ({100*(r['tok_a']-r['tok_b'])/max(r['tok_b'],1):.2f}%)")
        print(f"  over8192: base={r['over_b']} rule={r['over_a']} rescued={len(r['rescued'])} pushed_over={len(r['pushed_over'])}")
        print(f"  rescued: {r['rescued'][:12]}")
        print(f"  pushed_over: {r['pushed_over'][:12]}")
        if r['errs']: print(f"  errs: {len(r['errs'])} {r['errs'][:5]}")
    elif mode=='sound':
        rule=sys.argv[3] if len(sys.argv)>3 else 'hall'
        print("capturing solutions...")
        sols=capture_solutions(names)
        print(f"captured {len(sols)} solutions")
        r=run_sound(names, rule, sols)
        print(f"=== SOUND rule={rule} set={sel} puzzles={len(names)} ===")
        print(f"  fire_nodes={r['fire_nodes']} violations={len(r['violations'])}")
        for v in r['violations'][:20]: print("   ",v)

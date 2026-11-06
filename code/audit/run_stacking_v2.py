"""Stacking ablation under the final (converged) probe: audit/run_audit3.py protocol.
Arms: pretrained, classical, pretrained+classical (matched 64 dims), random0, random0+classical."""
import sys, json, copy, numpy as np, torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,"third_party/MtReNIRS/alignment/src")
from mtrenirs.alignment import align
from mtrenirs.models import MtReNIRS

d=np.load("data/luke_raw.npz",allow_pickle=True)
sig=d["signal"].astype(np.float32); y=d["labels"].astype(int)
subs=np.array([str(s) for s in d["subjects"]]); cht=[str(c) for c in d["ch_types"]]
r=align(positions=d["ch_pos"],names=[f"ch{i}" for i in range(len(d["ch_pos"]))],unit="m",modality="fnirs")
m=MtReNIRS.from_pretrained(checkpoint="third_party/MtReNIRS/models/weights/mtrenirs_10s.pt",device="cpu",
    ch_types=cht,ch_pos_1005=r.ch_pos_1005.astype(np.float32),ch_pos_1005_valid=r.valid.astype(bool))
def emb(model):
    o=[]
    with torch.no_grad():
        for i in range(0,len(sig),256): o.append(model.embed(torch.from_numpy(sig[i:i+256]),pool="flatten").cpu().numpy())
    E=np.concatenate(o,0); return E.reshape(len(E),-1)
def reinit(model,seed):
    c=copy.deepcopy(model); torch.manual_seed(seed)
    with torch.no_grad():
        for _,mm in c.encoder.named_modules():
            f=getattr(mm,"reset_parameters",None)
            if f is not None: f()
    return c
t=np.linspace(-1,1,sig.shape[2]); F=np.concatenate([sig.mean(2),(sig@t)/(t@t)],1)
E=emb(m); E0=emb(reinit(m,0))
def loso(blocks):
    acc=[]; n_pca=max(1,64//len(blocks))
    for s in np.unique(subs):
        tr,te=subs!=s,subs==s; a,b=[],[]
        for X in blocks:
            sc=StandardScaler().fit(X[tr]); A,B=sc.transform(X[tr]),sc.transform(X[te])
            p=PCA(n_components=min(n_pca,A.shape[0]-1,A.shape[1]),random_state=0).fit(A)
            TA=p.transform(A); sc2=StandardScaler().fit(TA)
            a.append(sc2.transform(TA)); b.append(sc2.transform(p.transform(B)))
        clf=LogisticRegression(max_iter=3000,C=1.0).fit(np.concatenate(a,1),y[tr])
        acc.append((clf.predict(np.concatenate(b,1))==y[te]).mean())
    return np.array(acc)
arms={"pretrained":[E],"classical":[F],"stack_pre_cls":[E,F],
      "random0":[E0],"stack_rnd_cls":[E0,F]}
R={k:loso(v) for k,v in arms.items()}
for k,v in R.items(): print(f"{k:15s} {v.mean():.4f} +/- {v.std():.4f}")
def paired(a,b,lab):
    d=R[a]-R[b]; t_,p=stats.ttest_rel(R[a],R[b])
    rng=np.random.default_rng(0); bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
    ci=np.percentile(bs,[2.5,97.5])
    print(f"{lab:28s} {d.mean()*100:+6.2f} pp p={p:.4f} CI=[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]")
    return {"diff_pp":float(d.mean()*100),"p":float(p),"ci95":[float(ci[0]*100),float(ci[1]*100)]}
out={"acc":{k:{"mean":float(v.mean()),"sd":float(v.std()),"folds":v.tolist()} for k,v in R.items()},"paired":{}}
print()
out["paired"]["stack - pretrained"]=paired("stack_pre_cls","pretrained","stack - pretrained")
out["paired"]["stack - classical"]=paired("stack_pre_cls","classical","stack - classical")
out["paired"]["stack_rnd - random0"]=paired("stack_rnd_cls","random0","stack_rnd - random0")
json.dump(out,open("results/results_stacking_v2.json","w"),indent=2)
print("saved results/results_stacking_v2.json")

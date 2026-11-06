"""Paired comparison: pretrained vs random-init vs classical, same folds.
Cheap re-run (no permutation null) to get per-fold accuracies for a paired test."""
import sys, json, copy, numpy as np, torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,"third_party/MtReNIRS/alignment/src")
from mtrenirs.models import MtReNIRS

d=np.load("data/luke_auditory.npz",allow_pickle=True); a=np.load("data/luke_aligned.npz")
sig=d["signal"].astype(np.float32); y=d["labels"].astype(int)
subs=np.array([str(s) for s in d["subjects"]]); cht=[str(c) for c in d["ch_types"]]
m=MtReNIRS.from_pretrained(checkpoint="third_party/MtReNIRS/models/weights/mtrenirs_10s.pt",device="cpu",
    ch_types=cht, ch_pos_1005=a["ch_pos_1005"].astype(np.float32), ch_pos_1005_valid=a["valid"].astype(bool))
def emb(model,pool="flatten"):
    out=[]
    with torch.no_grad():
        for i in range(0,len(sig),256):
            out.append(model.embed(torch.from_numpy(sig[i:i+256]),pool=pool).cpu().numpy())
    E=np.concatenate(out,0); return E.reshape(len(E),-1)
rnd=copy.deepcopy(m)
with torch.no_grad():
    for _,mod in rnd.encoder.named_modules():
        r=getattr(mod,"reset_parameters",None)
        if r is not None: r()
t=np.linspace(-1,1,sig.shape[2])
feats={"pretrained":emb(m),"random_init":emb(rnd),
       "classical":np.concatenate([sig.mean(2),(sig@t)/(t@t)],1)}
folds=[]
for s in np.unique(subs):
    tr,te=subs!=s,subs==s
    folds.append((tr,te))
def loso(X):
    acc=[]
    for tr,te in folds:
        sc=StandardScaler().fit(X[tr]); Xtr,Xte=sc.transform(X[tr]),sc.transform(X[te])
        p=PCA(n_components=64,random_state=0).fit(Xtr)
        clf=LogisticRegression(max_iter=300,C=1.0).fit(p.transform(Xtr),y[tr])
        acc.append((clf.predict(p.transform(Xte))==y[te]).mean())
    return np.array(acc)
R={k:loso(v) for k,v in feats.items()}
print("per-fold mean acc:")
for k,v in R.items(): print(f"  {k:12s} {v.mean():.4f} +/- {v.std():.4f}")
def paired(a,b,label):
    diff=R[a]-R[b]
    t,p=stats.ttest_rel(R[a],R[b]); w,pw=stats.wilcoxon(R[a],R[b])
    n=len(diff); bs=np.array([np.random.default_rng(i).choice(diff,n,replace=True).mean() for i in range(5000)])
    ci=np.percentile(bs,[2.5,97.5])
    print(f"{label:28s} diff={diff.mean()*100:+.2f} pp  t-p={p:.4f}  wilcoxon-p={pw:.4f}  bootstrap95%CI=[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] pp")
paired("pretrained","random_init","pretrained - random_init")
paired("pretrained","classical","pretrained - classical")
paired("random_init","classical","random_init - classical")
json.dump({k:v.tolist() for k,v in R.items()}, open("audit/paired_folds.json","w"), indent=2)

import sys, numpy as np, cv2
import os; sys.path.insert(0,'.'); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib.utils import cosilab
from board_fit import board_pose, OBJ
det={(int(c),int(i)):(tuple(s),np.asarray(u,float)) for c,i,s,u in np.load(sys.argv[1],allow_pickle=True)}
top={c:cosilab.load_camera(cosilab.DATA_ROOT,c) for c in (1,2,3,4)}
ds=[];depth=[];sizes={}
for (c,i),(s,u) in det.items(): sizes[s]=sizes.get(s,0)+1
print('detected sizes',sizes)
for i in range(332):
    seen=[c for c in top if (c,i) in det]
    if len(seen)<2: continue
    W={}
    for c in seen:
        s,u=det[(c,i)]; R,t=board_pose(u,s,top[c]['K'],top[c]['D']); Xc=OBJ[s]@R.T+t; depth.append(np.linalg.norm(Xc.mean(0)))
        W[c]=((Xc-top[c]['t_w2c'])@top[c]['R_w2c']).mean(0)
    for a in seen:
        for b in seen:
            if a<b: ds.append(np.linalg.norm(W[a]-W[b]))
ds=np.array(ds); print('pairwise centre distance m: p10 %.2f median %.2f p90 %.2f; <0.05 %d, <0.2 %d of %d'%(np.percentile(ds,10),np.median(ds),np.percentile(ds,90),(ds<.05).sum(),(ds<.2).sum(),len(ds)))
print('board distance from camera m: median %.2f p10 %.2f p90 %.2f'%(np.median(depth),np.percentile(depth,10),np.percentile(depth,90)))
# corner spacing in px for a sample
s,u=det[next(iter(det))]; print('sample size',s,'corner step px',np.linalg.norm(np.diff(u[:5],axis=0),axis=1).round(1))

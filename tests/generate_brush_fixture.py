"""Generate float64 oracle from the supplied, unmodified brush reference."""
import sys, json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'reference'))
from test_symptoms import BrushCanvas, make_brush
from fluid_core import Params
import color_model as cm
P=Params(); P.nx=160; P.ny=64
c=BrushCanvas(160,64,2,P); c.Rdry[:]=.995
def spectrum():
    result=c.Rdry.copy()
    for k in range(2):
        rl,tl=cm.layer_RT(c.a[k],np.maximum(c.beta[k],1e-9)[...,None],c.p[k][...,None],98000)
        result=np.where((c.d[k]>=P.eps_d)[...,None],cm.composite(rl,tl,result),result)
    return result
for sid, rgb in [(1,[30,60,200]),(2,[200,30,30])]:
    br=make_brush(12,rgb,1,P.h_stroke)
    for x in np.linspace(20,140,41):
        c.stamp(br,x,32,12,1,sid,x_dep=.1,x_pick=.1)
before=spectrum()[26:39,40:120].tolist()
stages=[c.d[:,32,:].tolist()]
grey=make_brush(12,[128,128,128],1,P.h_stroke)
grey['h'][:]=0; grey['p'][:]=0
for k in range(20):
    for x in np.linspace(40,120,27) if k%2==0 else np.linspace(120,40,27):
        c.stamp(grey,x,32,12,1,3+k,x_dep=.1,x_pick=.1,r_rep=0)
    stages.append(c.d[:,32,:].tolist())
R=c.Rdry.copy()
for k in range(2):
    rl,tl=cm.layer_RT(c.a[k],np.maximum(c.beta[k],1e-9)[...,None],c.p[k][...,None],98000)
    R=np.where((c.d[k]>=P.eps_d)[...,None],cm.composite(rl,tl,R),R)
out={'description':'Supplied float64 brush reference; s_sp=.25, x_dep=x_pick=.1 (supplied T-S1 scenario); 540 smudge stamps','stages':stages,'before':before,'patch':R[26:39,40:120].tolist(),'depth':c.d[:,26:39,40:120].tolist(),'ids':c.id[:,26:39,40:120].tolist()}
(root/'tests'/'brush-reference.json').write_text(json.dumps(out,separators=(',',':')),encoding='utf-8')
print('Generated float64 brush fixture: 13 x 80 x 38')




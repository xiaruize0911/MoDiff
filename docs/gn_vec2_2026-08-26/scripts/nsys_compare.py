import sqlite3, sys
from collections import defaultdict
def window(db):
    c=sqlite3.connect(db)
    rows=c.execute("""SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                      JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start""").fetchall()
    ah=[(s,e) for s,e,n in rows if 'gn_apply_delta_quantize_flat_vec2' in n]
    g=[ah[i+1][0]-ah[i][1] for i in range(len(ah)-1)]; m=sorted(g)[len(g)//2]
    b=[i for i,x in enumerate(g) if x>20*m]; lo,hi=ah[b[-1]+1][0],ah[-1][1]
    win=[(s,e,n) for s,e,n in rows if s>=lo and e<=hi]
    agg=defaultdict(lambda:[0.0,0])
    for s,e,n in win:
        agg[n.split('(')[0][:62]][0]+=(e-s)/1e6; agg[n.split('(')[0][:62]][1]+=1
    busy=0; last=lo; idle=0
    for s,e,n in sorted(win):
        if s>last: idle+=s-last
        last=max(last,e)
    return agg, sum(v[0] for v in agg.values()), (hi-lo)/1e6, idle/1e6

A,tA,wA,iA = window('int8_modiff.sqlite')
B,tB,wB,iB = window('int8_both.sqlite')
keys=set(A)|set(B)
print('%-60s %9s %9s %9s'%('kernel (ms/step)','before','after','delta'))
rowsp=[]
for k in keys:
    a=A.get(k,[0,0])[0]/20; b=B.get(k,[0,0])[0]/20
    if max(a,b)>0.25: rowsp.append((b-a,k,a,b))
for d,k,a,b in sorted(rowsp):
    print('%-60s %9.3f %9.3f %+9.3f'%(k,a,b,d))
print()
print('GPU busy   %.3f -> %.3f ms/step   (%+.3f)'%(tA/20,tB/20,(tB-tA)/20))

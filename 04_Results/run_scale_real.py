"""Scale probe on REAL instances beyond the exact range (n = NREQ in 20, 30): no exact frontier exists, so the reference is the
best-known energy over 3 long SA runs (100000 iters, seeds 2001-2003). Compared: one standard-budget run (20000 iters, seed 1001).
Arms LA_base, AUS_base; k in (2,4); T in (1e5 [slack], 3600). Output scale_real_n<NREQ>.csv.
Usage (from 03_Code_Execution): NREQ=20 NW=3 python -I ../04_Results/run_scale_real.py <draws>"""
import sys, os, csv, time
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","03_Code_Execution")); sys.path.insert(0, os.path.join(HERE,"..","03_Code_Execution","core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur import anneal_multi
TM=TimeModel(s_service=60.0,tau=180.0)
N=int(os.environ.get("NREQ","20"))
ARMS=[("LA_base","DLA4"),("AUS_base","DAU1")]
HF={"LA_base":lambda: height_la,"AUS_base":lambda: height_austin(0.0)}
CASES=[(2,1e5),(4,1e5),(2,3600.0),(4,3600.0)]
def one(a):
    (name,st_),seed=a
    from legs import LegCost
    from around import CityRaster
    city,nodes,reqs,meta=make_real_instance(st_,N,seed,3000.0,100.0,HF[name]())
    leg=CityLeg(LegCost(city,nodes,model="F",hw=TYPE_M600.hw,v=10.0,raster=CityRaster(city,res=10.0)),TYPE_M600)
    R=[Req(*r) for r in reqs]; rows=[]
    for k,T in CASES:
        t0=time.time(); s=anneal_multi(leg,R,TYPE_M600,TM,k,T,restarts=3,iters=20000,seed0=1001); ts=time.time()-t0
        L=[anneal_multi(leg,R,TYPE_M600,TM,k,T,restarts=3,iters=100000,seed0=2001+j) for j in range(3)]
        L=[x for x in L if x]
        best=min([x[0] for x in L]+([s[0]] if s else [])) if (L or s) else None
        rows.append(dict(arm=name,n=len(R),seed=seed,k=k,T=T,E_std=(round(s[0],3) if s else ""),mk_std=(round(s[1],1) if s else ""),
            E_best=(round(best,3) if best is not None else ""),n_long_found=len(L),E_long_worst=(round(max(x[0] for x in L),3) if L else ""),sec_std=round(ts,1)))
    return rows
if __name__=="__main__":
    draws=int(sys.argv[1]); nw=min(int(os.environ.get("NW","1")),3); t0=time.time()
    with Pool(nw) as p: res=p.map(one,[(a,1001+i) for a in ARMS for i in range(draws)],chunksize=1)
    rows=[r for rr in res for r in rr]
    out=os.path.join(HERE,os.environ.get("OUT",f"scale_real_n{N}.csv"))
    with open(out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote",out,len(rows),"sec",round(time.time()-t0))

import sys, os, time
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","03_Code_Execution")); sys.path.insert(0, os.path.join(HERE,"..","03_Code_Execution","core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur import anneal_multi
from legs import LegCost
from around import CityRaster
TM=TimeModel(s_service=60.0,tau=180.0)
n=int(sys.argv[1])
city,nodes,reqs,meta=make_real_instance("DLA4",n,1001,3000.0,100.0,height_la)
leg=CityLeg(LegCost(city,nodes,model="F",hw=TYPE_M600.hw,v=10.0,raster=CityRaster(city,res=10.0)),TYPE_M600)
R=[Req(*r) for r in reqs]
print(len(R),"reqs")
for k,T in ((2,1e5),(4,1e5),(4,3600),(4,2400)):
  for it,seed in ((20000,1001),(20000,1002),(100000,1003)):
    t0=time.time(); r=anneal_multi(leg,R,TYPE_M600,TM,k,T,restarts=3,iters=it,seed0=seed)
    print(k,T,it,seed,None if r is None else (round(r[0],1),round(r[1],1)),round(time.time()-t0,1),flush=True)

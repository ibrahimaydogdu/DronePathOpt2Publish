import sys,os,csv,time
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"..","03_Code_Execution"));sys.path.insert(0,os.path.join(HERE,"..","03_Code_Execution","core_d1"))
from drone3d_d2.realcity import make_real_instance,height_la
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600,TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_alns import alns
TM=TimeModel(s_service=60.0,tau=180.0)
rows=[r for r in csv.DictReader(open(os.path.join(HERE,"bridge_real_n12.csv"),encoding="utf-8")) if r["arm"]=="LA_base" and r["seed"]=="1001"]
from legs import LegCost
from around import CityRaster
city,nodes,reqs,meta=make_real_instance("DLA4",12,1001,3000.0,100.0,height_la)
leg=CityLeg(LegCost(city,nodes,model="F",hw=TYPE_M600.hw,v=10.0,raster=CityRaster(city,res=10.0)),TYPE_M600)
R=[Req(*r) for r in reqs]
for c in rows[:4]:
    for rs in (1001,1002):
        t0=time.time();r=alns(leg,R,TYPE_M600,TM,int(c["k"]),float(c["T"])+0.05,80000,rs)
        print(c["k"],c["theta"],"exact",c["E_exact"],"alns",None if r is None else round(r[0],3),"sa",c["E_heur"],round(time.time()-t0,1),"s",flush=True)

import sys,os,csv
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"..","03_Code_Execution"));sys.path.insert(0,os.path.join(HERE,"..","03_Code_Execution","core_d1"))
from drone3d_d2.realcity import make_real_instance,height_austin
from drone3d_d2.fleet import Req,trip_table,one_drone_table,fleet_frontier,min_energy_at
from drone3d_d2.dtypes import TYPE_M600,TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_alns import alns
TM=TimeModel(s_service=60.0,tau=180.0)
from legs import LegCost
from around import CityRaster
city,nodes,reqs,meta=make_real_instance("DAU1",12,1003,3000.0,100.0,height_austin(0.0))
leg=CityLeg(LegCost(city,nodes,model="F",hw=TYPE_M600.hw,v=10.0,raster=CityRaster(city,res=10.0)),TYPE_M600)
R=[Req(*r) for r in reqs]
h=one_drone_table(trip_table(leg,R,TYPE_M600,TM),12,TM)
F=fleet_frontier([h]*2,12); tmin=F[-1][1]
print("exact Tmin",tmin,"frontier head",[tuple(round(x,3) for x in p[:2]) for p in F[-3:]])
row=[r for r in csv.DictReader(open(os.path.join(HERE,"bridge_real_n12.csv"),encoding="utf-8")) if r["arm"]=="AUS_base" and r["seed"]=="1003" and r["k"]=="2" and float(r["theta"])==0][0]
print("stored T",row["T"])
for rs in (1004,):
    r=alns(leg,R,TYPE_M600,TM,2,float(row["T"])+0.05,80000,rs)
    print("alns E",r[0],"makespan",r[1],"exact E at stored T+0.05:",min_energy_at(F,float(row["T"])+0.05))

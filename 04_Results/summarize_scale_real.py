import csv,statistics as st
for n in (20,30):
    R=list(csv.DictReader(open(f"scale_real_n{n}.csv")))
    print("n",n,"rows",len(R))
    for arm in ("LA_base","AUS_base"):
        for k,T in ((2,1e5),(4,1e5),(2,3600.0),(4,3600.0)):
            S=[r for r in R if r["arm"]==arm and int(r["k"])==k and float(r["T"])==T]
            nf=sum(1 for r in S if r["E_std"]=="")
            g=[(float(r["E_std"])/float(r["E_best"])-1)*100 for r in S if r["E_std"]!="" and r["E_best"]!=""]
            sp=[(float(r["E_long_worst"])/float(r["E_best"])-1)*100 for r in S if r["E_long_worst"]!=""]
            print(arm,"k",k,"T",int(T),"notfound %d/%d"%(nf,len(S)),("gap med %.2f mean %.2f max %.2f"%(st.median(g),st.mean(g),max(g)) if g else ""),("| long-run spread med %.2f max %.2f"%(st.median(sp),max(sp)) if sp else ""),"| sec_std med %.1f"%st.median(float(r["sec_std"]) for r in S))

import csv,glob,sys,collections,statistics as st
N=sys.argv[1] if len(sys.argv)>1 else '20'
rows=[r for fn in sorted(glob.glob(f'large_n{N}/*.csv')) for r in csv.DictReader(open(fn,encoding='utf-8'))]
M=['SA_full','ALNS','ownABC','libABC']
cells=collections.defaultdict(list)
for r in rows: cells[(r['arm'],r['inst'],r['k'],r['theta'])].append(r)
best={k:min(float(r[m]) for r in v for m in M if r[m]!='') for k,v in cells.items() if any(r[m]!='' for r in v for m in M)}
print("n=%s rows %d cells %d (reference = best value found by any method/run in cell; not a bound)"%(N,len(rows),len(cells)))
print("method   noplan%  hit-best%(<=best+0.1%)  mean gap% (feasible)  median gap%")
for m in M:
    npn=sum(r[m]=='' for r in rows); hit=0; gaps=[]
    for r in rows:
        k=(r['arm'],r['inst'],r['k'],r['theta'])
        if r[m]!='' and k in best:
            g=(float(r[m])-best[k])/best[k]; gaps.append(g); hit+= g<=1e-3
    print(f"{m:8s} {100*npn/len(rows):6.1f} {100*hit/len(rows):10.1f} {100*sum(gaps)/max(1,len(gaps)):14.2f} {100*st.median(gaps):10.2f}")
print("by k and theta (hit-best%):")
for k in ('2','4'):
    for th in ('0.25','0.5','1.0'):
        s=[r for r in rows if r['k']==k and r['theta']==th]
        h={m:round(100*sum(1 for r in s if r[m]!='' and (float(r[m])-best[(r['arm'],r['inst'],r['k'],r['theta'])])/best[(r['arm'],r['inst'],r['k'],r['theta'])]<=1e-3)/len(s),1) for m in M}
        print(" k",k,"theta",th,h)
print("mean SA runtime s:",round(sum(float(r['sec_SA']) for r in rows)/len(rows),1))

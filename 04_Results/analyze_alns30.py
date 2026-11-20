import csv,glob,os
rd=lambda p:[r for fn in sorted(glob.glob(p)) for r in csv.DictReader(open(fn,encoding='utf-8'))]
A=rd('alns30/*.csv'); C={(r['arm'],r['inst'],r['k'],r['theta'],r['run']):r for r in rd('campaign30/*.csv')}
mt=lambda v,e: v!='' and float(v)<=float(e)+1e-3*max(1,abs(float(e)))
n=len(A);m=sum(mt(r['ALNS'],r['E_exact']) for r in A);np_=sum(r['ALNS']=='' for r in A)
below=sum(1 for r in A if r['ALNS']!='' and float(r['ALNS'])<float(r['E_exact'])-1e-3)
print("ALNS n=12, 72 cells x 30 runs: match %.1f%% noplan %.1f%% below-exact violations %d"%(100*m/n,100*np_/n,below))
for k in('SA_full','SA_plain','libABC','gABC'):
    print(" ",k,"%.1f%%"%(100*sum(mt(C[(r['arm'],r['inst'],r['k'],r['theta'],r['run'])][k],r['E_exact']) for r in A)/n))

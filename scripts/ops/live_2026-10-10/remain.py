import json,glob,re,os,sys
S=os.path.dirname(os.path.abspath(__file__)); T=S+'/transit10'
arms={'j':['C1','CY'],'e15':['C2','CX']}
for arm,boxes in arms.items():
    assigned={}; 
    for b in boxes:
        for ln in [l for p in (f'{S}/w_m_{b}.txt',f'{S}/w_r_{b}.txt') if os.path.exists(p) for l in open(p)]:
            for job in re.findall(r'CKE:(\S+?):\S+?:([\d,]+):',ln):
                for r in job[1].split(','): assigned[r]=(b,job[0])
    done=set(); inval=set()
    for f in glob.glob(f'{T}/*/bench2drive_out/mix_{arm}_*_x.json'):
        for r in json.load(open(f)).get('_checkpoint',{}).get('records',[]):
            m=re.search(r'RouteScenario_(\d+)',r['route_id']); rid=m.group(1)
            (inval if "set up" in r['status'] else done).add(rid)
    rem=[r for r in assigned if r not in done]
    print(arm,'assigned',len(assigned),'done',len(done&set(assigned)),'invalid(not redone)',len(inval-done),'remaining',len(rem))
    byjob={}
    for r in rem: byjob.setdefault(assigned[r],[]).append(r)
    for k,v in sorted(byjob.items()): print('  ',k,len(v),','.join(v[:12]))

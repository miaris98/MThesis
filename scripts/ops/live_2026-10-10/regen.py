import json,glob,re,os,sys,subprocess
S=os.path.dirname(os.path.abspath(__file__)); T=S+'/transit10'
CK={'j':'carla_armJ_ft_obst/model_epoch_020.pth','e15':'carla_armE_aug1_hires/model_epoch_015.pth'}
FL={'j':'WOR_SPEED_SCALE=0.8,WOR_NEWSVENDOR=q0=0.3;T=8;qmax=0.7','e15':'WOR_SPEED_SCALE=0.9,WOR_NEWSVENDOR=q0=0.3;T=8;qmax=0.7'}
arms={'j':['C1','CY'],'e15':['C2','CX']}
PORTS=[4400,4800,5200,5600]
for arm,boxes in arms.items():
    assigned={}
    for b in boxes:
        for p in (f'{S}/w_m_{b}.txt',f'{S}/w_r_{b}.txt'):
            if not os.path.exists(p): continue
            for ln in open(p):
                for job in re.findall(r'CKE:(\S+?):\S+?:([\d,]+):',ln):
                    for r in job[1].split(','): assigned[r]=job[0].endswith('_ep')
    done=set()
    for f in glob.glob(f'{T}/*/bench2drive_out/mix_{arm}_*_x.json'):
        for r in json.load(open(f)).get('_checkpoint',{}).get('records',[]):
            rid=re.search(r'RouteScenario_(\d+)',r['route_id']).group(1)
            if "set up" not in r['status']: done.add(rid)
    rem=sorted([r for r in assigned if r not in done], key=lambda r:-(3 if assigned[r] else 1))
    lanes=[(b,p,[],[]) for b in boxes for p in PORTS]; load=[0]*len(lanes)
    for r in rem:
        i=load.index(min(load)); lanes[i][2 if assigned[r] else 3].append(r); load[i]+=3 if assigned[r] else 1
    out={b:[] for b in boxes}
    for i,(b,p,ep,lo) in enumerate(lanes):
        jobs=[]
        if ep: jobs.append(f'CKE:mix_{arm}_{b}_{p}_rep:{CK[arm]}:{",".join(ep)}:{FL[arm]},CARLA_QUALITY=Epic')
        if lo: jobs.append(f'CKE:mix_{arm}_{b}_{p}_rlo:{CK[arm]}:{",".join(lo)}:{FL[arm]}')
        if jobs: out[b].append(f'{b} {p} 0 '+' '.join(jobs))
    for b in boxes:
        open(f'{S}/w_r_{b}.txt','w').write('\n'.join(out[b])+'\n')
    print(arm,'remaining',len(rem),'epic',sum(assigned[r] for r in rem),'loads',load)

import os, subprocess, sys, numpy as np
P='/tudelft.net/staff-umbrella/neon/cosilab_project'
def tag(path):
    out=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream_tags=timecode:stream=nb_frames','-of','csv=p=0',path],capture_output=True,text=True,check=True).stdout.split()
    nb,tc=out[0].split(',') if ',' in out[0] else (out[0],out[1])
    return tc,int(nb)
def tc2n(tc): h,m,s,f=map(int,tc.split(':')); return ((h*60+m)*60+s)*60+f
def n2tc(n): f=n%60; s=n//60; return f'{s//3600:02d}:{s//60%60:02d}:{s%60:02d}:{f:02d}'
def n2ts(n): f=n%60; s=n//60; return f'{s//3600:02d}:{s//60%60:02d}:{s%60:02d}.{round(f*1e6/60):06d}'
for c in ['01','02','03','04']:
    raw=[d for d in os.listdir(f'{P}/raw_sensor_data/gopro_data/top_view_camera-mingle_session_2') if d.startswith(f'camera_{c}')][0]
    raw=f'{P}/raw_sensor_data/gopro_data/top_view_camera-mingle_session_2/{raw}'
    T=f'{P}/4TU_upload/processed/videos/timestamps/mingle_session_2/camera_{c}-mingle_session_2'
    files=sorted(f for f in os.listdir(T) if f.endswith('.csv'))
    rows=[];orig=[];
    g0=None; res={'halfup':0,'halfeven':0,'halfdown':0}; tot=0; origbad=0; tagcont=0
    gidx=0; prev_end=None
    for f in files:
        tc,nb=tag(f'{raw}/{f[:-4]}.MP4')
        n0=tc2n(tc)
        if prev_end is not None and n0!=prev_end: tagcont+=1
        if g0 is None: g0=n0
        lines=open(f'{T}/{f}').read().split('\n')[1:]; lines=[l.split(',') for l in lines if l.strip()]
        assert len(lines)==nb,(f,len(lines),nb)
        g=gidx+np.arange(nb)
        for name,fn in [('halfup',None),('halfeven',None),('halfdown',None)]:
            w=g0+{'halfup':(g*1001+500)//1000,'halfeven':np.round(g*1001/1000).astype(int),'halfdown':(g*1001+499)//1000}[name]
            res[name]+=sum(n2tc(int(a))!=l[1] for a,l in zip(w,lines)); res[name+'_ts']=res.get(name+'_ts',0)+sum(n2ts(int(a))!=l[2] for a,l in zip(w,lines))
        origbad+=sum(n2tc(n0+i)!=l[0] for i,l in enumerate(lines))
        tot+=nb; gidx+=nb; prev_end=n0+nb
    print(c,len(files),tot,'orig mismatches',origbad,'tag discontinuities',tagcont,res,flush=True)

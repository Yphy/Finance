# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 01:07:51 2019

@author: sujih
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#수익률
data_raw=pd.ExcelFile('C:\\Users\\sujih\\Desktop\\4-2\\Financial_timeseries\\스마트 베타\\수익률3.xlsx')
tr=data_raw.parse('수익률2')

dates=tr.iloc[:,0].copy() #date 열
rest=tr.iloc[:,1:772].copy()

tau=12
mom_rest=(((rest/rest.shift(tau))-1)*100).copy()

nummom=len(mom_rest)
numcol=mom_rest.shape[1]
momrank=np.zeros([nummom,numcol])

#모멘텀 랭킹
for t in range(0,nummom):
 momrank[t,:]=list((mom_rest.iloc[t, :]).rank(ascending=False, method='min'))
#%% KBSTAR 모멘텀로우볼 
 #%% 36표준편차 rank
tau=36
vol=rest.rolling(tau).std()

volrank=np.zeros([nummom,numcol])

for t in range(0,nummom):
 volrank[t,:]=list((vol.iloc[t, :]).rank(ascending=True,method='min'))
#%% 결합
multi=momrank+volrank
multirank=pd.DataFrame(multi)
multirank=multirank.rank(axis=1, ascending=True, method='min')

#50 이상만 남김
m50=multirank.copy()
rank=50
m50[(m50>rank)]=0
m50[(m50)>0]=1
#%% 단순시총가중

#시가총액
marketcap=pd.ExcelFile('C:\\Users\\sujih\\Desktop\\4-2\\Financial_timeseries\\스마트 베타\\시총3.xlsx')
cap=marketcap.parse('시총2')

dates=tr.iloc[:,0].copy() #date 열
cap_r=cap.iloc[:,1:772].copy()

capnp=np.zeros([nummom,numcol])

for t in range(0,nummom):
 capnp[t,:]=cap_r.iloc[t,:]

#50개 시총
cap50=pd.DataFrame(capnp).mul(m50)

sumcap=sum(cap50, axis=1)
numdate=len(cap_r)
numstock=cap_r.shape[1]
wc=np.zeros([numdate,numstock])

for t in range(0,numdate):
  wc[t,:]=cap50.iloc[t,:]/sumcap[t]
#%% 시총 ceiling/  20% 제한, 20%넘는 비중은 넘는 자산 제외하고 다시 시총대로 배분
sumsumcap=sumcap.copy()  
wc_ceil=wc.copy()
ceill=0.2

for t in range(0, numdate):
 for i in range(0, numstock):
   if wc_ceil[t,i]>ceill:
      sumsumcap[t]=sumsumcap[t]-cap50.iloc[t,i]
      wc_ceil[t,i+1:]=wc_ceil[t,i+1:]+(wc[t,i]-ceill)*(cap50.iloc[t,i+1:]/sumsumcap[t]) 
      wc_ceil[t,i]=ceill 
      


#%%수익률
restnp=np.zeros([nummom,numcol])

for t in range(0,nummom):
 restnp[t,:]=rest.iloc[t,:]

rp=np.zeros([nummom,numcol])
for t in range(0,nummom):
 rp[t,:]=((wc_ceil[t-1,:]*restnp[t,:]))

rps=pd.DataFrame(rp).sum(axis=1)
#%%코스피
kraw= pd.ExcelFile('C:\\Users\\sujih\\Desktop\\4-2\\Financial_timeseries\\스마트 베타\\k200.xlsx')
kt=kraw.parse('k')

k200=kt.iloc[:,1].copy() 
k2r=(k200/k200.shift(1)-1)*100
#%%그림
dx=20021231
stdate2=pd.to_datetime(str(dx), format='%Y%m%d')
iddx=np.argmin(np.abs(dates-stdate2))
datedx=dates[iddx:].copy()
rpdx=(rps[iddx:].copy()).reset_index(drop=True)


vp=pd.Series(np.ones([nummom-iddx])*100)
for t in range(1,nummom-iddx):
    vp[t]=vp[t-1]*(1+rpdx[t]/100)


vk=pd.Series(np.ones([nummom-iddx])*100)
for t in range(1,nummom-iddx):
    vk[t]=vk[t-1]*(1+k2r[t]/100)


    
vpmdd=(vp/vp.cummax()-1)*100
vkmdd=(vk/vk.cummax()-1)*100
#%% 손절매, 해도 좋고 안해도 좋고
point=-10
vpcut=vp.copy()

for t in range(1, nummom-iddx):
 if vpmdd[t-1]<=point:
    vpcut[t]=vk[t] #그냥 코스피 넣는데 다른 걸로 대체가능



#%% 그래프 그리기 
import matplotlib.pyplot as plt
from matplotlib import gridspec
fig = plt.figure(figsize=(10, 7))   # figsize = (가로길이, 세로길이)
gs = gridspec.GridSpec(nrows=2,     # row 개수 
                       ncols=1,     # col 개수 
                       height_ratios=[8, 3], 
                       width_ratios=[5])  # subplot의 크기를 서로 다르게 설정

ax0 = plt.subplot(gs[0])
ax0.plot(datedx, vp, datedx, vk, 'r-')

ax0.grid(True)
ax0.legend(labels=('vol_mom','k200'))

ax1 = plt.subplot(gs[1])
ax1.plot(datedx, vpmdd, datedx, vkmdd, 'r-')

ax1.grid(True)
ax1.legend(labels=('mdd_vol_mom','mdd_k200'), loc='lower right')

plt.show()

#%% KBSTAR 모멘텀밸류 pbr+momentum
#pbr
pbr_raw=pd.ExcelFile('C:\\Users\\sujih\\Desktop\\4-2\\Financial_timeseries\\스마트 베타\\pbr.xlsx')
pb=pbr_raw.parse('pbr')
dates_pb=pb.iloc[:,0].copy() 
restpb=pb.iloc[:,1:772].copy()

numpb=len(restpb)
numcolpb=restpb.shape[1]
pbrank=np.zeros([numpb,numcol])
#pbr 낮을 수록 높은 랭킹
for t in range(0,numpb):
 pbrank[t,:]=list((restpb.iloc[t, :]).rank(ascending=True, method='min'))
#%% 결합
#모멘텀 날짜가 길어서 자름
dx=20000131
momslicedate=pd.to_datetime(str(dx), format='%Y%m%d')
mmmiddx=np.argmin(np.abs(dates-momslicedate))
momrank2=momrank[mmmiddx:].copy()


multi2=momrank2+pbrank
multirank2=pd.DataFrame(multi2)
multirank2=multirank2.rank(axis=1, ascending=True, method='min')

m50_2=multirank2.copy()
rank=50
m50_2[(m50_2>rank)]=0
m50_2[(m50_2)>0]=1

#%%시총
capnp2=capnp[mmmiddx:].copy()
cap50_2=pd.DataFrame(capnp2).mul(m50_2)

sumcap2=sum(cap50_2, axis=1)
numdate2=len(cap50_2)
numstock2=cap50_2.shape[1]
wc2=np.zeros([numdate2,numstock2])

for t in range(0,numdate2):
  wc2[t,:]=cap50_2.iloc[t,:]/sumcap2[t]
#%% 시총 ceiling/  20% 제한
sumsumcap2=sumcap2.copy()  
wc_ceil2=wc2.copy()
ceill=0.2

for t in range(0, numdate2):
 for i in range(0, numstock2):
   if wc_ceil2[t,i]>ceill:
      sumsumcap2[t]=sumsumcap2[t]-cap50_2.iloc[t,i]
      wc_ceil2[t,i+1:]=wc_ceil2[t,i+1:]+(wc2[t,i]-ceill)*(cap50_2.iloc[t,i+1:]/sumsumcap2[t]) 
      wc_ceil2[t,i]=ceill 
      

  
  
  
  
#%% 수익률

restnp2=np.zeros([numpb,numcolpb])
rest2=rest[mmmiddx:].copy()

for t in range(0,numpb):
 restnp2[t,:]=rest2.iloc[t,:]

rp2=np.zeros([numpb,numcolpb])
for t in range(0,numpb):
 rp2[t,:]=((wc_ceil2[t-1,:]*restnp2[t,:]))

rps2=pd.DataFrame(rp2).sum(axis=1)
#%%그림
dx=20021231
stdate2=pd.to_datetime(str(dx), format='%Y%m%d')
iddx=np.argmin(np.abs(dates-stdate2))
datedx=dates[iddx:].copy()

iddxpb=np.argmin(np.abs(dates_pb-stdate2))

rpdx2=(rps2[iddxpb:].copy()).reset_index(drop=True)


vp2=pd.Series(np.ones([numpb-iddxpb])*100)
for t in range(1,numpb-iddxpb):
    vp2[t]=vp2[t-1]*(1+rpdx2[t]/100)


vk=pd.Series(np.ones([nummom-iddx])*100)
for t in range(1,nummom-iddx):
    vk[t]=vk[t-1]*(1+k2r[t]/100)


    
vp2mdd=(vp2/vp2.cummax()-1)*100
vkmdd=(vk/vk.cummax()-1)*100

#%% 그래프 그리기 
import matplotlib.pyplot as plt
from matplotlib import gridspec
fig = plt.figure(figsize=(10, 7))   # figsize = (가로길이, 세로길이)
gs = gridspec.GridSpec(nrows=2,     # row 개수 
                       ncols=1,     # col 개수 
                       height_ratios=[8, 3], 
                       width_ratios=[5])  # subplot의 크기를 서로 다르게 설정

ax0 = plt.subplot(gs[0])
ax0.plot(datedx, vp2, datedx, vk, 'r-')

ax0.grid(True)
ax0.legend(labels=('value_mom','k200'))

ax1 = plt.subplot(gs[1])
ax1.plot(datedx, vp2mdd, datedx, vkmdd, 'r-')

ax1.grid(True)
ax1.legend(labels=('mdd_value_mom','mdd_k200'), loc='lower right')

plt.show()
















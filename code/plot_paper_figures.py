from config import JD
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def binary_curve(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);ix=np.argsort(-p,kind='stable');y=y[ix];p=p[ix]
    ends=np.r_[np.where(np.diff(p)!=0)[0],len(p)-1]
    tp=np.cumsum(y)[ends];fp=ends+1-tp
    return tp,fp,p[ends],int(y.sum()),len(y)-int(y.sum())
def roc_curve(y,p):
    tp,fp,th,pos,neg=binary_curve(y,p)
    return np.r_[0,fp/neg],np.r_[0,tp/pos],np.r_[np.inf,th]
def roc_auc_score(y,p):
    f,t,_=roc_curve(y,p);return float(np.trapezoid(t,f))
def precision_recall_curve(y,p):
    tp,fp,th,pos,neg=binary_curve(y,p)
    return np.r_[1,tp/(tp+fp)],np.r_[0,tp/pos],th
def average_precision_score(y,p):
    pr,rc,_=precision_recall_curve(y,p);return float(np.sum(np.diff(rc)*pr[1:]))

R=Path(JD)/'results'; O=Path(JD)/'figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'pdf.fonttype':42,
 'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
C=['#2466A2','#D57831','#168A72','#9B75A6']; models=['cox_multivar','lightgbm','mosaic']
MN={'rate_only':'Duration only','cox_crude':'Univariable Cox','cox_multivar':'Multivariable Cox','cloglog':'Cloglog',
 'logit':'Logistic','lightgbm':'LightGBM','xgboost':'XGBoost','random_forest':'Random forest','icboost':'ICBoost','mosaic':'MOSAIC'}
DS={'caries':'Caries','gingivitis':'Gingivitis','combined':'Combined outcome'}
PS={'cv':'Repeated CV','loyo':'Leave-one-year-out','split':'Chronological test'}
tasks={'audiometria_final':'Audiometry status','gingivitis_incidencia':'Gingivitis','pterigion_incidencia':'Pterygium','depresion_final':'Depression status',
 'estereopsis_incidencia':'Stereopsis','PatronRestrictivo_final':'Restrictive pattern','PatronObstructivo_final':'Obstructive pattern',
 'lumbalgia_final':'Low back pain','onicomicosis_incidencia':'Onychomycosis','discromatopsia_incidencia':'Dyschromatopsia',
 'infarto_final':'Myocardial infarction','acv_final':'Cerebrovascular event','ametropia_incidencia':'Ametropia','cardiovascular_final':'Cardiovascular status',
 'estres_final':'Stress status','ansiedad_final':'Anxiety status'}
ABL={'A1_no_aux':'No auxiliary supervision','A2_aux_equal_weights':'All task weights = 1','A3_aux_budget_equal':'Equal weights within budget',
 'A4_aux_uncertainty_weights':'Uncertainty weighting','A5_binary_head':'Binary head','A6_cox_head':'Cox head','A7_mlp_trunk':'No second-order interactions',
 'A8_attn_trunk':'Attention backbone','A9_no_ema':'No EMA','A10_single_model':'Single neural model','A11_no_distil':'No distillation',
 'A12_no_boost':'No tree branches','A13_no_aux_features':'No auxiliary tree features','A14_no_stack':'Equal logit fusion',
 'A15_no_calibration':'No final calibration','A16_no_rex':'No year regularization'}
raw=pd.read_csv(R/'experiments/main_raw.csv')
oof=pd.read_csv(R/'experiments/oof_predictions.csv')
def grid(ax):ax.grid(alpha=.18);ax.set_axisbelow(True)
def save(fig,n,name):
    p=O/f'fig{n:02d}_{name}'
    for ext in ['pdf','png']:fig.savefig(p.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.12,dpi=190)
    plt.close(fig);print(p.name)
def legend(fig,ax,ncol=3,y=0):
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False)
def contrast_label(r):
    dic={'Manual/Physical':'Manual work','Customer service/Social services':'Customer/social services','Healthcare professional':'Healthcare',
     '1 to <5 years':'Tenure 1 to <5 years','5 to <10 years':'Tenure 5 to <10 years','≥10 years':'Tenure >=10 years','Male':'Male'}
    blocks={'Smoker':'Smoking','Alcohol consumption':'Alcohol use','Night work':'Night work','Age in years':'Age','Abdominal waist':'Waist circumference',
     'Systolic blood pressure':'Systolic BP','Diastolic blood pressure':'Diastolic BP','Glucose':'Glucose','Triglycerides':'Triglycerides','Total cholesterol':'Total cholesterol'}
    if r.label=='Yes':return blocks[r.block]+' (yes)'
    if r.label=='per 5-unit increase':return blocks[r.block]+' (per 5 units)'
    return dic.get(r.label,r.label)

# 1 All original numeric cells retained as a data-integrity check.
d=pd.read_csv(R/'data_check.csv').dropna(subset=['value_ours'])
fig,axs=plt.subplots(1,2,figsize=(8.7,3.7),layout='constrained')
for lab,m,c in [('Counts and events',d.stat.isin(['n','events']),C[0]),('Person-years',d.stat=='py',C[1]),('Other statistics',~d.stat.isin(['n','events','py']),C[2])]:
 s=d[m];axs[0].scatter(s.value_paper,s.value_ours,s=12,facecolors='none',edgecolors=c,label=f'{lab}({len(s)})')
axs[0].plot([.3,4e4],[.3,4e4],'--',color='grey',lw=.8)
axs[0].set(xscale='log',yscale='log',xlabel='Published statistic',ylabel='Computed statistic',title='(a) Summary agreement');axs[0].legend(fontsize=7,frameon=False)
d['grp']=np.where(d.block=='Years of exposure','Tenure groups','Other variables')
piv=d.groupby(['table','grp']).agrees.agg(['sum','count'])
for j,(g,c) in enumerate([('Other variables',C[2]),('Tenure groups',C[1])]):
 vals=[100*piv.loc[(t,g),'sum']/piv.loc[(t,g),'count'] for t in range(1,5)]
 x=np.arange(4)+(j-.5)*.38;axs[1].bar(x,vals,width=.34,label=g,color=c)
 for xx,v,t in zip(x,vals,range(1,5)):axs[1].text(xx,v+2,f'{v:.0f}%\n(n={piv.loc[(t,g),"count"]})',ha='center',fontsize=7)
axs[1].set(xticks=range(4),xticklabels=['Source table 1','Source table 2','Source table 3','Source table 4'],ylim=(0,124),ylabel='Agreement under original rules (%)',title='(b) Agreement by table and group');axs[1].legend(loc='center right',frameon=False)
for a in axs:grid(a)
save(fig,1,'data_integrity')

# 2 Calendar composition.
ry=pd.read_csv(R/'era/rate_by_year.csv');ec=pd.read_csv(R/'era/era_composition.csv')
fig,axs=plt.subplots(1,3,figsize=(10,3.5),layout='constrained',gridspec_kw={'width_ratios':[1.3,1,1]})
s=ry[(ry.dataset=='caries')&(ry.n>=100)].sort_values('year')
for i,r in enumerate(s.itertuples()):axs[0].errorbar(i,r.rate,yerr=[[r.rate-r.lo],[r.hi-r.rate]],fmt='o',capsize=2,color=C[1] if r.year==2015 else C[0])
axs[0].set(xticks=range(len(s)),xticklabels=s.year.astype(int),ylabel='Detection rate / 1,000 PY',xlabel='Baseline year',ylim=(0,50),title='(a) Caries detection rate');axs[0].tick_params(axis='x',rotation=45)
for a,block,labs,title in [(axs[1],'Years of exposure',['<1 year','1 to <5 years','5 to <10 years','>=10 years'],'(b) Tenure groups'),(axs[2],'Type of occupation',['Office','Manual work','Customer/social services','Healthcare'],'(c) Occupational groups')]:
 s=ec[ec.block==block];a.barh(range(4),s.pct_2015_baseline,color=C[0]);a.set(yticks=range(4),yticklabels=labs,xlim=(0,62),xlabel='Entered in 2015 (%)',title=title);a.invert_yaxis()
 for i,v in enumerate(s.pct_2015_baseline):a.text(v+1,i,f'{v:.1f}',va='center',fontsize=8)
for a in axs:grid(a)
save(fig,2,'year_distribution')

# 3 Crude and penalized/year-adjusted descriptive estimates.
cv=pd.read_csv(R/'era/crude_vs_adjusted.csv');fig,axs=plt.subplots(1,3,figsize=(10.2,6.2),sharey=True)
for a,(ds,lab) in zip(axs,DS.items()):
 s=cv[cv.dataset==ds].reset_index(drop=True);y=np.arange(len(s))
 for prefix,off,c,marker,label in [('crude',-.15,C[0],'o','Unadjusted'),('adj',.15,C[1],'s','Year-adjusted + regularized')]:
  a.errorbar(s[prefix+'_hr'],y+off,xerr=[s[prefix+'_hr']-s[prefix+'_lo'],s[prefix+'_hi']-s[prefix+'_hr']],fmt=marker,color=c,markersize=3.2,lw=.8,label=label)
 a.axvline(1,color='grey',ls='--',lw=.8);a.set(xscale='log',xlim=(.03,12),title=lab,xlabel='Cox HR (log scale)');grid(a)
axs[0].set(yticks=np.arange(17),yticklabels=[contrast_label(r) for r in cv[cv.dataset=='caries'].itertuples()]);axs[0].invert_yaxis()
fig.subplots_adjust(left=.29,right=.99,top=.94,bottom=.13,wspace=.14);legend(fig,axs[0],2,.01);save(fig,3,'year_associations')

# 4 Window AUC versus stratified AUC.
s=raw[(raw.dataset=='caries')&(raw.protocol=='cv')].groupby('model')[['auc','strat_auc']].mean().reindex(MN)
fig,ax=plt.subplots(figsize=(7,4.5));y=np.arange(len(s))
ax.hlines(y,s.strat_auc,s.auc,color='#BFCAD4',lw=2)
ax.scatter(s.auc,y,color=C[1],label='Window AUC');ax.scatter(s.strat_auc,y,color=C[0],label='Follow-up-stratified AUC')
ax.axvline(.5,ls='--',color='grey',lw=.8);ax.set(yticks=y,yticklabels=[MN[m] for m in s.index],xlabel='AUC',xlim=(.44,.91));ax.invert_yaxis();grid(ax)
fig.subplots_adjust(left=.22,bottom=.2);legend(fig,ax,2,.02);save(fig,4,'window_auc')

# 5 Main comparison, all models and all protocols.
order=[m for m in MN if m!='rate_only'];fig,axs=plt.subplots(3,3,figsize=(10,8.4),sharex=True,sharey=True)
for i,(p,pl) in enumerate(PS.items()):
 for j,(ds,dl) in enumerate(DS.items()):
  a=axs[i,j];s=raw[(raw.dataset==ds)&(raw.protocol==p)].groupby('model').strat_auc.agg(['mean','std','count']).reindex(order)
  for y,(m,r) in enumerate(s.iterrows()):
   se=0 if np.isnan(r['std']) else r['std']/np.sqrt(r['count']);a.errorbar(r['mean'],y,xerr=se,fmt='o',color=C[2] if m=='mosaic' else C[0],ms=4,capsize=2,lw=.9)
  a.axvline(.5,ls='--',color='grey',lw=.8);a.set(xlim=(.38,.95),yticks=range(len(order)),ylim=(len(order)-.5,-.5));grid(a)
  if j==0:a.set_yticklabels([MN[m] for m in order]);a.set_ylabel(pl,labelpad=9)
  if i==0:a.set_title(dl)
  if i==2:a.set_xlabel('Follow-up-stratified AUC')
fig.subplots_adjust(left=.21,right=.99,bottom=.08,top=.95,wspace=.12,hspace=.22);save(fig,5,'all_models')

# 6 ROC / PR. Short legends in reserved space, no overlaid long model names.
fig,axs=plt.subplots(2,3,figsize=(9.2,7.1))
rocstats=[]
for j,(ds,lab) in enumerate(DS.items()):
 sub=oof[(oof.dataset==ds)&(oof.protocol=='cv')]
 for k,m in enumerate(models):
  s=sub[sub.model==m];fpr,tpr,_=roc_curve(s.y,s.p);pr,rc,_=precision_recall_curve(s.y,s.p)
  auc=roc_auc_score(s.y,s.p);ap=average_precision_score(s.y,s.p)
  axs[0,j].plot(fpr,tpr,color=C[k],label=f'{MN[m]}  {auc:.3f}')
  axs[1,j].plot(rc,pr,color=C[k],label=f'{MN[m]}  {ap:.3f}')
  rocstats.append({'dataset':ds,'model':m,'pooled_auc':auc,'pooled_ap':ap})
 for i in range(2):
  a=axs[i,j];a.set(xlim=(0,1),ylim=(0,1.02));grid(a)
  a.legend(loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=8,ncol=1,handlelength=1.6)
 axs[0,j].plot([0,1],[0,1],ls='--',color='grey',lw=.8);axs[0,j].set(title=lab,xlabel='False positive rate')
 prev=sub[sub.model==models[0]].y.mean();axs[1,j].axhline(prev,ls='--',color='grey',lw=.8);axs[1,j].set(xlabel='Recall')
 axs[1,j].text(.97,.09,f'Prevalence {prev:.3f}',ha='right',fontsize=8,transform=axs[1,j].transAxes)
axs[0,0].set_ylabel('True positive rate');axs[1,0].set_ylabel('Precision')
fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.16,hspace=.9,wspace=.24);save(fig,6,'roc_pr')
pd.DataFrame(rocstats).to_csv(O.parent/'pooled_curve_metrics.csv',index=False)

# 7 Calibration.
fig,axs=plt.subplots(1,3,figsize=(9.3,3.5))
for a,(ds,lab) in zip(axs,DS.items()):
 hi=.05
 for k,m in enumerate(models):
  s=oof[(oof.dataset==ds)&(oof.protocol=='cv')&(oof.model==m)].copy();s['bin']=pd.qcut(s.p,10,duplicates='drop')
  g=s.groupby('bin',observed=True)[['p','y']].mean();a.plot(g.p,g.y,'o-',color=C[k],ms=3,label=MN[m]);hi=max(hi,g.max().max())
 a.plot([0,hi*1.08],[0,hi*1.08],'--',color='grey',lw=.8);a.set(xlim=(0,hi*1.08),ylim=(0,hi*1.08),title=lab,xlabel='Mean predicted probability');grid(a)
axs[0].set_ylabel('Observed positive proportion');fig.subplots_adjust(left=.09,right=.99,top=.9,bottom=.24,wspace=.3);legend(fig,axs[0],3,.015);save(fig,7,'calibration')

# 8 Decision curves.
fig,axs=plt.subplots(1,3,figsize=(9.3,3.6));ths=np.linspace(.005,.2,80)
for a,(ds,lab) in zip(axs,DS.items()):
 for k,m in enumerate(models):
  s=oof[(oof.dataset==ds)&(oof.protocol=='cv')&(oof.model==m)];n=len(s)
  vals=[((s.y.eq(1)&s.p.ge(t)).sum()-(s.y.eq(0)&s.p.ge(t)).sum()*t/(1-t))/n for t in ths]
  a.plot(ths,vals,color=C[k],label=MN[m])
 prev=s.y.mean();a.plot(ths,prev-(1-prev)*ths/(1-ths),'--',color='grey',label='Refer all');a.axhline(0,color='black',lw=.7,label='Refer none')
 a.set(title=lab,xlabel='Risk threshold',ylim=(-.01,max(.012,prev*1.1)));grid(a)
axs[0].set_ylabel('Net benefit per individual');fig.subplots_adjust(left=.09,right=.99,top=.9,bottom=.26,wspace=.32);legend(fig,axs[0],5,.015);save(fig,8,'decision_curves')

# 9 Ablation.
d=pd.read_csv(R/'experiments/ablation_raw.csv').groupby(['protocol','model']).strat_auc.mean();order=list(ABL)
fig,axs=plt.subplots(1,2,figsize=(10.2,6),sharey=True)
for a,p in zip(axs,['cv','loyo']):
 delta=d[p].reindex(order)-d[p]['MOSAIC'];a.barh(range(16),delta,color=[C[1] if v<0 else C[2] for v in delta])
 for y,v in enumerate(delta):a.text(v+(.002 if v>=0 else -.002),y,f'{v:+.4f}',va='center',ha='left' if v>=0 else 'right',fontsize=7.5)
 a.axvline(0,color='grey',lw=.8);a.set(xlim=(-.16,.03),title=PS[p],xlabel='sAUC change from full model');grid(a)
axs[0].set(yticks=range(16),yticklabels=[ABL[k] for k in order]);axs[0].invert_yaxis();fig.subplots_adjust(left=.36,right=.99,bottom=.12,top=.93,wspace=.14);save(fig,9,'ablation')

# 10 Auxiliary weights. Removed misleading unit-weight line and fixed horizontal scale.
d=pd.read_csv(R/'experiments/task_weights.csv').sort_values('weight');fig,ax=plt.subplots(figsize=(8.8,5.6))
ax.barh(range(len(d)),d.weight,color=C[2],height=.68)
for y,r in enumerate(d.itertuples()):ax.text(r.weight+.001,y,f'{r.weight:.3f}  (n={r.events:,})',va='center',fontsize=8)
ax.set(yticks=range(len(d)),yticklabels=[tasks[k] for k in d.task],xlim=(0,.077),xlabel='Mean auxiliary task weight');grid(ax)
fig.subplots_adjust(left=.34,right=.98,bottom=.12,top=.97);save(fig,10,'auxiliary_weights')

# 11 Year-specific validation.
s=raw[(raw.dataset=='caries')&(raw.protocol=='loyo')];years=sorted(s.fold.unique());fig,ax=plt.subplots(figsize=(7.5,3.8))
for k,m in enumerate(models):
 d=s[s.model==m].set_index('fold').loc[years];ax.plot(range(len(years)),d.strat_auc,'o-',color=C[k],label=MN[m])
ax.axhline(.5,color='grey',ls='--',lw=.8);ax.set(xticks=range(len(years)),xticklabels=[x[1:] for x in years],xlabel='Held-out baseline year',ylabel='Follow-up-stratified AUC',ylim=(.35,.86));grid(ax)
fig.subplots_adjust(bottom=.25,left=.12,right=.99,top=.97);legend(fig,ax,3,.02);save(fig,11,'year_validation')

# 12 Neural cumulative profile and empirical label residual coverage.
hz=pd.read_csv(R/'experiments/hazard_profile.csv');cov=pd.read_csv(R/'experiments/conformal_coverage.csv')
fig,axs=plt.subplots(1,2,figsize=(9,3.8));seq=['#ADCBE5','#679ACA','#3476B4','#134E83']
for i,(g,s) in enumerate(hz.groupby('risk_group')):axs[0].step(s.t_end,100*s.cum_risk,where='post',color=seq[i],label=['Q1 (low)','Q2','Q3','Q4 (high)'][i])
axs[0].set(xlabel='Time since baseline (years)',ylabel='Neural cumulative probability (%)',title='(a) Predicted-score quartiles');axs[0].legend(frameon=False,fontsize=8)
axs[1].bar(range(5),100*cov.coverage,color=C[2]);axs[1].axhline(90,color='grey',ls='--');axs[1].set(xticks=range(5),xticklabels=[f'Q{x}' for x in range(1,6)],xlabel='Observation-duration quintile',ylabel='Empirical label coverage (%)',ylim=(0,105),title='(b) Residual coverage')
for i,v in enumerate(cov.coverage):axs[1].text(i,100*v+2,f'{100*v:.1f}',ha='center',fontsize=8)
for a in axs:grid(a)
fig.tight_layout();save(fig,12,'risk_coverage')

# 13 Permutation importance. Keep identifiers alongside translated common variables.
d=pd.read_csv(R/'experiments/importance.csv').groupby('feature').drop_strat_auc.mean().sort_values(ascending=False).head(18)
feat={'base_year':'Baseline year','edad1':'Age','pas1':'Systolic BP','pad1':'Diastolic BP','cintura1':'Waist circumference','glucosa1':'Glucose',
 'trigliceridos1':'Triglycerides','colesterol1':'Total cholesterol','imc1':'BMI','sexo':'Sex','fuma':'Smoking','alcohol':'Alcohol use',
 'hemoglobina1':'Hemoglobin','tiempomeses1':'Baseline tenure (months)','espirometria1':'Spirometry','diabetes1':'Baseline diabetes',
 'cadera1':'Hip circumference','occ_2':'Occupation category 2','occ_3':'Occupation category 3','familiar_hta':'Family history of hypertension','familiar_dm':'Family history of diabetes','audiometria1':'Baseline audiometry'}
fig,ax=plt.subplots(figsize=(9,5.8));y=np.arange(len(d));ax.barh(y,d.values,color=C[0]);ax.invert_yaxis()
for yy,v in zip(y,d.values):ax.text(v+.001,yy,f'{v:.4f}',va='center',fontsize=8)
ax.set(yticks=y,yticklabels=[feat.get(k,k) for k in d.index],xlabel='sAUC decrease after permutation',xlim=(0,d.max()*1.23));grid(ax)
fig.subplots_adjust(left=.36,right=.99,bottom=.12,top=.98);save(fig,13,'importance');print('IMPORTANCE',d.to_dict())

# 14 Learning curves.
d=pd.read_csv(R/'experiments/learning_curve.csv');fig,ax=plt.subplots(figsize=(7.2,3.8))
for k,m in enumerate(['mosaic','A1_no_aux']):
 g=d[d.model==m].groupby('frac').strat_auc.agg(['mean','std','count']);se=g['std']/np.sqrt(g['count']);x=g.index.to_numpy()*100
 ax.plot(x,g['mean'],'o-',color=C[k],label='MOSAIC' if m=='mosaic' else 'No auxiliary supervision');ax.fill_between(x,g['mean']-se,g['mean']+se,color=C[k],alpha=.15)
ax.set(xlabel='Training sample fraction (%)',ylabel='Follow-up-stratified AUC');grid(ax)
fig.subplots_adjust(bottom=.25,left=.13,right=.99,top=.97);legend(fig,ax,2,.02);save(fig,14,'learning_curve')

# 15 Feature sets. Dot plot preserves values without a truncated bar baseline.
d=pd.read_csv(R/'experiments/featureset_raw.csv');fig,ax=plt.subplots(figsize=(7.8,4))
for k,m in enumerate(models):
 g=d[d.model==m].groupby('featureset').strat_auc.agg(['mean','std','count']).reindex(['paper','base','traj']);x=np.arange(3)+(k-1)*.15
 ax.errorbar(x,g['mean'],yerr=g['std']/np.sqrt(g['count']),fmt='o',capsize=3,color=C[k],label=MN[m])
 for xx,v in zip(x,g['mean']):ax.text(xx,v+.014,f'{v:.3f}',ha='center',fontsize=8)
ax.set(xticks=range(3),xticklabels=['Basic baseline (18)','Full baseline (52)','With post-baseline data (114)'],ylabel='Follow-up-stratified AUC',ylim=(.57,.96));grid(ax)
fig.subplots_adjust(bottom=.25,left=.12,right=.98,top=.96);legend(fig,ax,3,.02);save(fig,15,'feature_sets')
print('Finished',len(list(O.glob('*.pdf'))),'figures')

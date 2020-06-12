import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xg




  
#Read Traing Data and Convert it to Numpy Array
with open("train.csv") as file:
        data=csv.reader(file)
        a=list(data)
 #Remove the First Coulmn and Row form the Dataset   
a=np.asarray(a)
a=a[1:]
a=a[:,1:]
le=a.shape
row=le[0]
col=le[1]   
      
#Convert any Non Numiric value to zeros in Numiric-Valued Coulmns      
k=0
for i in range(0,col):
        c=0
        x=np.array([])
        inx=0
        for ii in range(0,row):
            temp=a[ii,i]
            temp1=list(map(ord,temp))
            temp1=np.asarray(temp1)
            temp1=temp1-48
            if np.any(temp1>9):
                if c==0:
                    c=1
                    x=np.array([temp])
                    inx=ii
                else:
    
                    x=np.append(x,temp)
                    inx=np.append(inx,ii)
    
        if x.size==row:
            if k==0:
                y=x.reshape((row,1))
                k=9
                inn=i
            else:
                y=np.block([[y,x.reshape((row,1))]])
                inn=np.append(inn,i)
        else :
            if x.size>0:
                a[inx,i]=0;
                
            
            
            
    

#Convert Non Numeric coulmns to Numiric Values Using OneHotEncoder    
ct = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore'),inn)],remainder=StandardScaler(with_mean=False, with_std=False))
z=ct.fit_transform(a)
z=z.toarray()
        
    
out=z[:,-1];
out=out.reshape((row,1))
z=z[:,:-1]
inxx1=np.zeros((row,1))
s=np.random.default_rng();
ts=np.floor(.7*row);
ts=ts.astype('int');
val=s.integers(0,row-1,(ts,1))
inxx1=inxx1.astype('bool')
inxx1[val]=True
inxx2=np.invert(inxx1)
    
dtrain= xg.DMatrix(z[inxx2[:,0]],out[inxx2[:,0]]);
dtest= xg.DMatrix(z[inxx1[:,0]],out[inxx1[:,0]]);
param = { 'eta':.3, 'silent':1, 'objective':'reg:squarederror','max_depth':6}
    
evallits=[(dtrain,'train'),(dtest,'eval')]
    
num_round = 15
ss=xg.cv(param,dtrain,nfold=10)
print(ss)
bst = xg.train(param, dtrain, num_round,evallits, early_stopping_rounds=10,verbose_eval=False)
bst.save_model('0001.model')
outt=out[inxx1[:,0]]
outtes=np.block([[bst.predict(dtest)],[outt[:,0]]])
outtes.tolist()






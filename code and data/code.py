import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

df=pd.read_csv('BlackFriday.csv')

###data cleaning
df=df.dropna()
df_freq=df.iloc[:,:2]
###data illustration
df['Gender'].unique()
##gender
explode = (0.1,0)  
fig1,ax1=plt.subplots(figsize=(12,7))
plt.pie(df['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.savefig('Gender_pie.jpeg')
plt.figure(figsize=(12,7))
sns.boxplot(x='Gender',y='Purchase',data=df)
plt.savefig('Gender_boxplot.jpeg')

##age
fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(df['Age'],hue=df['Gender'])
plt.savefig('age_bar.jpeg')

##city
explode = (0.1, 0, 0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df.groupby('City_Category')['Purchase'].sum(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.savefig('city_pie.jpeg')

fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(df['City_Category'],hue=df['Age'])
plt.savefig('citybyage_bar.jpeg')

##material
#label=['Underage 0-17','Retired +55','Middleage 26-35','46-50 y/o','Oldman 51-55','Middleage+ 36-45','Youth']
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['Marital_Status'].value_counts(),explode=explode, labels=['Yes','No'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.savefig('marerial_pie.jpeg')



###classification
from scipy import interp
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB                                                                                                
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.metrics import roc_curve,auc
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

##Predict gender of a customer based on the Product_Category_1,Product_Category_2,Product_Category_3
df_new=df
del df_new['User_ID']
del df_new['City_Category']
df_new['Gender'].replace('M',0,inplace=True)
df_new['Gender'].replace('F',1,inplace=True)
#0-17:1,18-25:2,26-35:3,36-45:4,46-50:5,51-55:6,55+:7
df_new['Age'].replace('0-17',1,inplace=True)
df_new['Age'].replace('18-25',2,inplace=True)
df_new['Age'].replace('26-35',3,inplace=True)
df_new['Age'].replace('36-45',4,inplace=True)
df_new['Age'].replace('46-50',5,inplace=True)
df_new['Age'].replace('51-55',6,inplace=True)
df_new['Age'].replace('55+',7,inplace=True)
df_new['Stay_In_Current_City_Years'].replace('4+',4,inplace=True)
X=df_new.iloc[:,6:9]
le=LabelEncoder()
Y=df['Gender']

clf_dt=DecisionTreeClassifier()
clf_nb=GaussianNB()
pipe_dt=Pipeline([['sc',StandardScaler()],['clf',clf_dt]])
pipe_nb=Pipeline([['sc',StandardScaler()],['clf',clf_nb]])                                                                                              
                                                                                                                                                          
clf_labels=['DecisionTree','Naive Bayes']                                                                                           
                                                                                                                                                          
mean_tpr_all=0.0                                                                                                                                          
mean_fpr_all=np.linspace(0, 1, 100)                                                                                                                       
###10-fold cross validation                                                                                                                               
plt.figure(figsize=(10,8))
for clf,label in zip([pipe_dt,pipe_nb],clf_labels):                                                                              
        #scores=cross_val_score(estimater=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')                                                                
        cv=StratifiedKFold(Y,n_folds=10)                                                                                                             

        mean_tpr=0.0                                                                                                                                      
        mean_fpr=np.linspace(0, 1, 100)                                                                                                                   
        for i,(train,test) in enumerate(cv): 
                clf.fit(np.array(X)[train],np.array(Y)[train])

                prob_predict_y = clf.predict_proba(np.array(X)[test])

                predictions_validation = prob_predict_y[:, 1]

                fpr, tpr, _ = roc_curve(np.array(Y)[test], predictions_validation)

                roc_auc=auc(fpr,tpr)

                mean_tpr+=interp(mean_fpr,fpr,tpr)
                mean_tpr[0]=0.0
        mean_tpr /=10
        mean_tpr[-1]=1.0
        mean_auc=auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr,lw=2,label=' %s (area = %0.2f)' % (label,mean_auc))                                                                    
                                                                                                                                                          
        mean_tpr_all+=interp(mean_fpr_all,mean_fpr,mean_tpr)                                                                                              
        mean_tpr_all[0]=0                                                                                                                                 
# plot line x=y  
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6))
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('10-folds cross validation')
plt.legend(loc="lower right")
plt.savefig('AUC_curve_10_folds.jpeg')

###frequent items productA-->productB
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

###select product_id and user_id colums df_freq
products_by_users = dict((k, v.values) for k, v in df_freq.groupby("User_ID")["Product_ID"])
df_freq=products_by_users.values()
df_freq_data=[]
for d in df_freq:
	df_freq_data.append(list(d.tolist()))

te=TransactionEncoder()
te_ary = te.fit(df_freq_data).transform(df_freq_data)
da = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(da, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, min_threshold=0.1)
pd.DataFrame.to_csv(rules,"Rules.csv")

###clusters= age category by product category
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

#label={1:'Underage 0-17',7:'Retired +55',3:'Middleage 26-35',5:'46-50 y/o',6:'Oldman 51-55',4:'Middleage+ 36-45',2:'Youth'}

RS=100
km=KMeans(n_clusters=7,random_state=RS)
X=df_new.iloc[:1000,6:9].values
kms=km.fit(X)
digits_proj=TSNE(random_state=RS).fit_transform(X)
txts=[]
def scatter(x,colors):
	palette=np.array(sns.color_palette("hls",7))
	f=plt.figure(figsize=(8,8))
	ax=plt.subplot(aspect='equal')
	sc=ax.scatter(x[:,0],x[:,1],lw=0,s=40,c=palette[colors.astype(np.int)])
	plt.xlim(-50,50)
	plt.ylim(-50,50)
	
	T=[]	
	for i in range(7):
		xtext,ytext=np.median(x[colors==i,:],axis=0)
		txt=ax.text(xtext,ytext,str(i+1),fontsize=10)
		txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"),PathEffects.Normal()])
		T.append(txt)
	return f, ax, sc, T

#scatter(digits_proj,y-1)
#plt.savefig('Tsne_cluster_actual.jpeg')
kms_labels=kms.labels_
scatter(digits_proj,kms_labels)
plt.savefig('Tsne_cluster_kms.jpeg')





import pandas as pd
import numpy as np
import re

dx = ['M10','M100','M109',#gout
		'M051','M053','M058','M059','M06','M060','M061','M062','M063','M064','M068','M069',#rheumatoid
		'I10','I500',#ht
		'E102','E109','E110','E111','E112','E113','E114','E115','E116','E117','E118','E119',#dm
		'E780','E781','E782','E783','E784','E785','E786','E787','E788',#dlp
		'I500','I501','I509',#chf
		'I20','I200','I201','I208','I209','I210','I211','I212','I214','I219','I240','I250','I251','I252','I255','I259',#ischemic heart disease
		'S320','S321','S322',#vertebral fx
		'S420',#clavicle fx
		'S422','S423','S424','S429',#humerus fx
		'S525','S526',#wrist fx
		'S323','S324','S325','S327','S328',#pelvis fx
		'S720','S721','S722','S723','S724','S728','S729',#hip fx
		'S82','S821','S822','S823','S824','S825','S826','S828',#leg fx
		'nan'
		]

drug_list = [	'BUDE01',#glucoc_budesonide
					'HYDI02',#glucoc_cortisone
					'LODI01','DEXT04','DEXI04','DEXT05','DEXI07',#glucoc_dexa
					'PRET05','PRET08',#glucoc_prednisolone
					'CALT05','CALT07','CALT09',#calcium
					'CALT01','CALT10',#calcium+vit d
					'CALT08','OSST03','VITT10','ALPT01','MEDT01','ONET01','ONET02','BONT04',#vit d
					'MYCE02','MIAE01','MIAI01','CALI04',#calcitonin
					'GLAT02',#menatetrenone vit k2
					'FORI02','MEGI01',#teriparatide daily
					'CELT05',#raloxifene daily
					'BONT02',#ibandronate daily
					'OSTI01',#ibandronate 3 months
					'PROL02',#stromtium daily
					'ALET01','FOST02',#alendronate weekly
					'FOST03','FOST05',#alendronate + vit d weekly
					'ACTT06','REST07',#risedronate weekly
					'ACTT07',#risedronate monthly
					'PROI11',#denosumab 6 month
					'ZOMI01','ZINI02','ACLI01','ZOLI03'#zoledronic yearly
					]
def onehot_demo(prefix=''):
	demofull = pd.read_csv(prefix+'demo_data.csv', index_col=0)
	demofull.rename(columns={'date': 'reg_date'}, inplace=True)
	demofull['icd10'] = demofull['icd10'].apply(lambda x: str(x)[:4])

	demo = demofull[demofull['icd10'].isin(dx)]
	icd10 = demo[['txn','icd10']]
	icd10 = pd.get_dummies(icd10,prefix='icd10')
	icd10 = icd10.groupby(['txn']).sum()
	icd10 = icd10.reset_index()
	demo.drop(columns=['icd10'],inplace=True)
	result = pd.merge(demofull, icd10, on=['txn'] ,how='left')
	if 'icd10_nan' in result.columns:
		result.drop(columns=['icd10_nan'],inplace=True)
	result.drop(columns=['icd10','dx_type'],inplace=True)
	result = result.drop_duplicates()
	print(result)
	result.to_csv(prefix+'demo_onehot.csv')

def onehot_drug(prefix=''):
	drug = pd.read_csv(prefix+'drug_data.csv', index_col=0)
	drug.rename(columns={'date': 'drug_date','code':'drug_code','name':'drug_name'}, inplace=True)
	drug = drug[drug['drug_code'].isin(drug_list)]
	drug.drop(columns=['drug_name','tradename'],inplace=True)
	drug_code = drug[['txn','drug_code']]
	drug_code = pd.get_dummies(drug_code,prefix='drug')
	drug_code = drug_code.groupby(['txn']).sum()
	drug_code = drug_code.reset_index()
	drug.drop(columns=['drug_code'],inplace=True)
	result = pd.merge(drug, drug_code, on=['txn'] ,how='inner')
	print(result)
	result.to_csv(prefix+'drug_onehot.csv')

def onehot_lab(prefix=''):
	lab = pd.read_csv(prefix+'lab_data.csv', index_col=0)
	lab.rename(columns={'date': 'lab_date','code':'lab_code','name': 'lab_item_name'}, inplace=True)
	lab.drop(columns=['lab_code','lab_name'],inplace=True)
	lab['lab_item_name'] = lab['lab_item_name'].apply(lambda x: re.sub('[^a-z]','',str(x).lower()))
	lab['value'] = lab['value'].apply(lambda x: str(x).split(' ')[0])
	lab_code = lab[['txn','lab_item_name']]
	lab_code = pd.get_dummies(lab_code,prefix='lab')
	for c in lab_code.columns:
		if c != 'txn':
			lab_code[c] = lab_code[c]*lab['value']
			lab_code[c] = lab_code[c].apply(lambda x: re.sub('nan','',str(x).lower())[:4])
			lab_code[c] = pd.to_numeric(lab_code[c], errors='coerce')
	lab_code = lab_code.groupby(['txn']).sum()
	lab_code = lab_code.reset_index()
	lab.drop(columns=['lab_item_name','value'],inplace=True)
	result = pd.merge(lab, lab_code, on=['txn'] ,how='inner')
	print(result)
	result.to_csv(prefix+'lab_onehot.csv')



def combine(prefix=''):

	bmd = pd.read_csv(prefix+'os_data.csv', index_col=0)
	bmd.rename(columns={'date': 'bmd_date','name':'bmd_name'}, inplace=True)

	demo_onehot = pd.read_csv(prefix+'demo_onehot.csv', index_col=0)
	drug_onehot = pd.read_csv(prefix+'drug_onehot.csv', index_col=0)
	lab_onehot = pd.read_csv(prefix+'lab_onehot.csv', index_col=0)

	result = pd.merge(demo_onehot, drug_onehot, on=['txn'] ,how='outer')
	result = pd.merge(result, lab_onehot, on=['txn'] ,how='outer')
	result = pd.merge(result, bmd, on=['txn'] ,how='outer')

	result['date'] = result['reg_date'].combine_first(result['drug_date'])
	result['date'] = result['date'].combine_first(result['lab_date'])
	result['date'] = result['reg_date'].combine_first(result['bmd_date'])
	result.drop(columns=['reg_date','drug_date','lab_date','bmd_date'],inplace=True)
	result = result.drop_duplicates()
	result.replace(0, np.nan, inplace=True)
	result = result[result['hn'].notna()]
	print(result)

	result.to_csv(prefix+'result.csv')

def create_dataset(prefix):
	onehot_demo(prefix=prefix)
	onehot_drug(prefix=prefix)
	onehot_lab(prefix=prefix)
	combine(prefix=prefix)

#create_dataset('')
#create_dataset('i')

def create_bmd_dataset(prefix):
	df = pd.read_csv(prefix+'result.csv',index_col=0)
	df['year'] = df['date'].apply(lambda x: str(x[:4]))
	df.drop(columns=['txn','date'], inplace=True)
	df = df.groupby(['hn','year','bmd_name']).first()
	for c in df.columns:
		if c[:5] == 'icd10' or c[:4] == 'drug':
			df[c] = df[c].apply(lambda x: 1 if x >=1 else 0)
	df.to_csv(prefix+'result_year.csv')
	print(df)

'''
result = pd.read_csv('result.csv', index_col=0)
r = result[['drug_code','drug_name']].groupby(['drug_code','drug_name']).size()
r.to_csv('drug_list.csv')
r = result[['drug_code']].groupby(['drug_code']).size()
r.to_csv('drugcode_list.csv')
print(r)
'''

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error	 
from sklearn.metrics import mean_absolute_error	 
from sklearn.metrics import mean_squared_error	 
from sklearn.metrics import mean_squared_log_error	 
from sklearn.metrics import median_absolute_error	 
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def validate(clfs,df,t):
	testset = df.sample(frac=0.25, random_state=16)
	X_test = testset.drop(columns=[t])
	y_test = testset[t]
	trainingset = df[~df.index.isin(testset.index)]
	X_train = trainingset.drop(columns=[t])
	y_train = trainingset[t]
	for clf in clfs:
		clf.fit(X_train,y_train)
		pred = clf.predict(X_test)
		print(clf.__class__.__name__)
		print('mean absolute error: '+str(mean_absolute_error(y_test, pred)))
		print('mean square error: '+str(mean_squared_error(y_test, pred)))
		print('r2 score: '+str(r2_score(y_test, pred)))
		corr, _ = pearsonr(y_test, pred)
		print('Pearsons correlation: %.3f' % corr)

		testset[clf.__class__.__name__.lower()] = pred
		#scores = cross_val_score(clf, X, y, cv=10)
		#print(clf)
		#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		'''
		import matplotlib.pyplot as plt
		import numpy as np
		import pandas as pd

		n = 5
		f = 100
		fig, axs = plt.subplots(n)
		fig.suptitle(clf)
		for i in range(n):
			axs[i].plot( range(f), y_test[i*f:(i+1)*f], marker='o', markerfacecolor='green', markersize=5, color='black', linewidth=1, label="actual "+t)
			axs[i].plot( range(f), pred[i*f:(i+1)*f], marker='o', markerfacecolor='red', markersize=5, color='black', linewidth=1, label="predicted "+t)

		plt.legend()
		axs[0].text(0,0.9,"mean absolute error: %0.2f" % (mean_absolute_error(y_test, pred)))
		axs[0].text(20,0.9,"mean square error: %0.2f" % (mean_squared_error(y_test, pred)))
		axs[0].text(40,0.9,"r2 score: %0.2f" % (r2_score(y_test, pred)))
		plt.show()
		'''
		'''
		import matplotlib.pyplot as plt
		import seaborn as sns
		sns.regplot(x=y_test, y=pred)
		plt.ylim(-5, 2)
		plt.xlim(-5, 2)
		plt.text(-4.8,1.5,"mean absolute error: %0.2f" % (mean_absolute_error(y_test, pred)))
		plt.text(-4.8,1.1,"mean square error: %0.2f" % (mean_squared_error(y_test, pred)))
		plt.text(-4.8,0.7,"r2 score: %0.2f" % (r2_score(y_test, pred)))
		plt.text(-4.8,0.3,"Pearsons correlation: %0.2f" % (corr))
		plt.title(clf)
		plt.xlabel('actual '+t)
		plt.ylabel('predicted '+t)
		plt.show()
		'''
	print(testset)
	testset.to_csv('predicted_'+t+'.csv')
	
	
def train(df,t):
	

	from sklearn import tree
	from sklearn import svm
	from sklearn import linear_model
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.naive_bayes import GaussianNB
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.linear_model import LinearRegression
	clfs = [	tree.DecisionTreeRegressor(),
				svm.SVR(),
				#linear_model.SGDRegressor(max_iter=1000),
				KNeighborsRegressor(n_neighbors=10),
				GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=0),
				RandomForestRegressor(random_state=0, n_estimators=100, max_depth=5),
				LinearRegression()]

	
	print(t)
	validate(clfs,df,t)

'''		
t = 'femur_bmd'
df = pd.read_csv('result.csv',index_col=0)
df = df[df[t].notna()]
sex = pd.get_dummies(df['sex'],prefix='sex')
df['m'] = sex['sex_m']
df['f'] = sex['sex_f']
df = df.fillna(0)
print(df)

df = df.drop(columns=[	'hn','txn',
								'femur_t_score','femur_z_score',
								'lumbar_bmd','lumbar_t_score','lumbar_z_score',
								'radius_bmd','radius_t_score','radius_z_score',
								'bmd_name','date','sex'
								])
train(df,t)
'''

def train_femur_tscore():
	odf = pd.read_csv('result_year.csv')
	idf = pd.read_csv('iresult_year.csv')
	df = odf.append(idf)
	t = 'femur_t_score'
	df = df[df[t].notna()]
	sex = pd.get_dummies(df['sex'],prefix='sex')
	df['m'] = sex['sex_m']
	df['f'] = sex['sex_f']
	df['prev1_'+t] = df.groupby(['hn'])[t].transform(lambda x:x.shift())
	#No result difference between 1 or n previous bmd
	#df['prev2_'+t] = df.groupby(['hn'])[t].transform(lambda x:x.shift(2))
	df = df.fillna(0)
	df = df.drop(columns=[	'hn',
									'femur_bmd','femur_z_score',
									'lumbar_bmd','lumbar_t_score','lumbar_z_score',
									'radius_bmd','radius_t_score','radius_z_score',
									'bmd_name','year','sex'
									])
	train(df,t)

def train_lumbar_tscore():
	odf = pd.read_csv('result_year.csv')
	idf = pd.read_csv('iresult_year.csv')
	df = odf.append(idf)
	t = 'lumbar_t_score'
	df = df[df[t].notna()]
	sex = pd.get_dummies(df['sex'],prefix='sex')
	df['m'] = sex['sex_m']
	df['f'] = sex['sex_f']
	df['prev1_'+t] = df.groupby(['hn'])[t].transform(lambda x:x.shift())
	df = df.fillna(0)
	df = df.drop(columns=[	'hn',
									'femur_bmd','femur_t_score','femur_z_score',
									'lumbar_bmd','lumbar_z_score',
									'radius_bmd','radius_t_score','radius_z_score',
									'bmd_name','year','sex'
									])
	train(df,t)

def fx_history(x):
	v_fx = ['icd10_S320','icd10_S321','icd10_S322'] #vertebral fx
	non_v_fx = ['icd10_S420',#clavicle fx
					'icd10_S422','icd10_S423','icd10_S424','icd10_S429',#humerus fx
					'icd10_S525','icd10_S526',#wrist fx
					'icd10_S323','icd10_S324','icd10_S325','icd10_S327','icd10_S328',#pelvis fx
					'icd10_S720','icd10_S721','icd10_S722','icd10_S723','icd10_S724','icd10_S728','icd10_S729',#hip fx
					'icd10_S82','icd10_S821','icd10_S822','icd10_S823','icd10_S824','icd10_S825','icd10_S826','icd10_S828',#leg fx
					]
	
	for f in v_fx:
		if x[f] > 0:
			return 'fx'
	for f in non_v_fx:
		if x[f] > 0:
			return 'fx'
	return 'no_fx'
	'''
	fx = 'no_fx'
	for f in v_fx:
		if x[f] > 0:
			fx = 'v_fx'
			break
	for f in non_v_fx:
		if x[f] > 0 and fx == 'v_fx':
			fx = 'both_fx'
			break
		elif  x[f] > 0:
			fx = 'non_v_fx'
			break
	return fx
	'''
def train_fx():
	odf = pd.read_csv('result_year.csv')
	idf = pd.read_csv('iresult_year.csv')
	df = odf.append(idf)
	testset = df.copy()
	t = 'fx_history'
	sex = pd.get_dummies(df['sex'],prefix='sex')
	df['m'] = sex['sex_m']
	df['f'] = sex['sex_f']
	df[t] = df.apply(fx_history, axis=1)
	df = df.fillna(0)
	df = df.drop(columns=[	'hn',
									'icd10_S320','icd10_S321','icd10_S322',#vertebral fx
									'icd10_S420',#clavicle fx
									'icd10_S422','icd10_S423','icd10_S424','icd10_S429',#humerus fx
									'icd10_S525','icd10_S526',#wrist fx
									'icd10_S323','icd10_S324','icd10_S325','icd10_S327','icd10_S328',#pelvis fx
									'icd10_S720','icd10_S721','icd10_S722','icd10_S723','icd10_S724','icd10_S728','icd10_S729',#hip fx
									'icd10_S82','icd10_S821','icd10_S822','icd10_S823','icd10_S824','icd10_S825','icd10_S826','icd10_S828',#leg fx
									'bmd_name','year','sex'
									])

	from sklearn import tree
	from sklearn import svm
	from sklearn import linear_model
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.naive_bayes import GaussianNB
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.linear_model import LinearRegression

	from sklearn.neural_network import MLPClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.svm import SVC
	from sklearn.gaussian_process import GaussianProcessClassifier
	from sklearn.gaussian_process.kernels import RBF
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
	from sklearn.ensemble import GradientBoostingClassifier
	clfs = [	KNeighborsClassifier(4),
				#SVC(kernel="linear", C=0.025),
				#SVC(gamma=2, C=1),
				#GaussianProcessClassifier(1.0 * RBF(1.0)),
				DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
				RandomForestClassifier(max_depth=5, n_estimators=100, class_weight='balanced'),
				MLPClassifier(alpha=1, max_iter=1000),
				AdaBoostClassifier(),
				GaussianNB(),
				#QuadraticDiscriminantAnalysis(),
				GradientBoostingClassifier()]
	
	testset = df.sample(frac=0.25, random_state=16)
	X_test = testset.drop(columns=[t])
	y_test = testset[t]
	trainingset = df[~df.index.isin(testset.index)]
	X_train = trainingset.drop(columns=[t])
	y_train = trainingset[t]

	for clf in clfs:
		clf.fit(X_train,y_train)
		preds = clf.predict_proba(X_test)
		'''
		predictions = clf.predict(X_test)	
		print(clf)
		from sklearn.metrics import classification_report, confusion_matrix
		print("Confusion Matrix:")
		print(confusion_matrix(y_test, predictions))

		print("Classification Report")
		print(classification_report(y_test, predictions))
		'''
		i = 0
		for c in clf.classes_:
			name = clf.__class__.__name__.lower()+'_'+c
			testset[name] = preds[:,i]
			i = i+1
	print(testset)
	testset.to_csv('fx_prediction.csv')
#create_bmd_dataset('')
#create_bmd_dataset('i')
#r2 score ~ 0.3 - 0.4
#train_femur_tscore()
#r2 score ~ 0.2 - 0.3
#train_lumbar_tscore()
train_fx()


















  

import mysql.connector as mysql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re

path = '../../secret/data/'

sql_path = '../../secret/sql/'

def convert(x):
    try:
        return x.encode('latin-1','replace').decode('tis-620','replace')
    except AttributeError:
        return x

def decode(df):
	for c in df.columns:
		if df[c].dtype == 'object':
			df[c] = df[c].apply(convert)
	return df

def clean_sex(x):
	x = str(x).replace(' ','')
	if x == 'ช':
		return 'm'
	elif x == 'ญ':
		return 'f'
	else:
		return 0

def clean_age(x):
	if x > 150:
		return 0
	else:
		return x

def get_raw_data(filename,config):
	file = Path(path+'raw/'+filename+'.csv')
	if file.is_file():
		return pd.read_csv(path+'raw/'+filename+'.csv', index_col=0)
	else:
		db_connection = get_connection(config)
		df = pd.read_sql(getquery(filename), con=db_connection)
		db_connection.close()
		df.to_csv(path+'raw/'+filename+'.csv')
		return df

def get_connection(config):
	return mysql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
def getquery(filename):
	f = open(sql_path+filename+'.sql', 'r')
	sql = f.read()
	f.close()
	return sql

def get_lumbar_bmd(x):
	x = x.lower()
	x = re.sub('\\\\[^ ]* ','',x)
	for i in x.split('\n'):
		if re.search('\d\. ',i) != None:
			r = re.search('l\d.*l?\d.*(\d\.\d\d\d)',i)
			b = ''
			t = ''
			if r == None:
				r = re.search('l\d.*l?\d.*=.*(\d\d\d)',i)
				if r == None:
					r = re.search('l\d.*=.*(\d\.\d\d\d)',i)
					if r != None:
						b = r.group()
						t = re.findall('([\+\-]\d+\.\d+) *s\.?d\.?',i)
				else:
					b = r.group()
					t = re.findall('([\+\-]\d+\.\d+) *s\.?d\.?',i)
			else:
				b = r.group()
				t = re.findall('([\+\-]\d+\.\d+) *s\.?d\.?',i)

			if b != '':
				if len(t) < 1:
					t = ';'
				elif len(t) == 1:
					t = t[0]+';'
				else:
					t = t[0]+';'+t[1]
				t = re.sub('\+','',t)
				r = re.search('\d\.\d+',b)
				if r != None:
					return(r.group()+';'+t)
				else:
					r = re.search('\d\.?\d+',b)
					if r != None:
						s = r.group()
						if s.startswith('0'):
							return(s[0]+'.'+s[1:]+';'+t)
						else:
							return('0.'+s+';'+t)
	return ';;'

def get_femur_bmd(x):
	x = x.lower()
	x = re.sub('\\\\[^ ]* ','',x)
	for i in x.split('\n'):
		if re.search('\d\. ',i) != None:
			r = re.search('femo?u?r.*=.*(\d\.\d\d\d?)',i)
			b = ''
			t = ''
			if r == None:
				r = re.search('fem.*=?.*([01]\.* ?\d\d\d?)[^ \W]',i)
				if r != None:
					b = r.group()
				t = re.findall('([\+\-]\d+\.\d+) *s\.?d\.?',i)
			else:
				b = r.group()
				t = re.findall('([\+\-]\d+\.\d+) *s\.?d\.?',i)

			if b != '':
				b = re.sub('\.+','.',b)
				b = re.sub('\. +','.',b)
				if len(t) < 1:
					t = ';'
				elif len(t) == 1:
					t = t[0]+';'
				else:
					t = t[0]+';'+t[1]
				t = re.sub('\+','',t)
				r = re.search('(\d\.\d+)',b)
				if r != None:
					return(r.group()+';'+t)
				else:
					r = re.search('\d+',b)
					if r != None:
						s = r.group()
						return(s[0]+'.'+s[1:]+';'+t)

	return ';;'

def get_radius_bmd(x):
	x = x.lower()
	x = re.sub('\\\\[^ ]* ','',x)
	for i in x.split('\n'):
		if re.search('\d\. ',i) != None:
			r = re.search('radius.*=.*(\d\.\d\d\d?)',i)
			b = ''
			t = ''
			if r != None:
				b = r.group()
				t = re.findall('([\+\-]\d+\.\d+) *s\.?d\.?',i)
			if b != '':
				b = re.sub('\.+','.',b)
				b = re.sub('\. +','.',b)
				if len(t) < 1:
					t = ';'
				elif len(t) == 1:
					t = t[0]+';'
				else:
					t = t[0]+';'+t[1]
				t = re.sub('\+','',t)
				r = re.search('(\d\.\d+)',b)
				if r != None:
					return(r.group()+';'+t)
	return ';;'

def get_data(config):

	#table = ['demo_data','idemo_data','lab_data','ilab_data','drug_data','idrug_data','os_data','ios_data']
	table = ['demo_data']
	for filename in table:
		df = get_raw_data(filename,config)
		df = decode(df)

		if 'sex' in df:
			df['sex'] = df['sex'].apply(clean_sex)
		if 'age' in df:	
			df['age'] = df['age'].apply(clean_age)

		if filename == 'lab_data' or filename == 'ilab_data':
			d1 = df['name'].str.split(';',expand=True)
			d1 = d1.merge(df, right_index = True, left_index = True)
			d1 = d1.melt(id_vars = ['txn','date','code','lab_name','value'], value_name = 'name')
			d1 = d1.sort_values(['txn','date','code','variable'], ascending=True)
			d1 = d1.drop('value', axis=1)
			d1 = d1[d1['variable'] != 'name']
			d2 = df['value'].str.split(';',expand=True)
			d2 = d2.merge(df, right_index = True, left_index = True)
			d2 = d2.melt(id_vars = ['txn','date','code','lab_name','name'], value_name = 'value')
			d2 = d2.sort_values(['txn','date','code', 'variable'], ascending=True)
			d2 = d2.drop('name', axis=1)
			d2 = d2[d2['variable'] != 'name']
			df = pd.merge(d1,d2,on=['txn','date','code','lab_name','variable'])
			df = df[['txn','date','code','lab_name','name','value']]
			df = df[(df['name'].str.contains('Hb'))
					| (df['name'].str.contains('Total Ca'))
					| (df['name'].str.contains('Inorganic P'))
					| (df['name'].str.contains('Albumin'))
					| (df['name'].str.contains('AST (GOT)'))
					| (df['name'].str.contains('ALT (GPT)'))
					| (df['name'].str.contains('Alkaline Phosphatase'))
					| (df['name'].str.contains('BUN'))
					| (df['name'].str.contains('Creatinine'))
					| (df['name'].str.contains('TSH'))
					| (df['name'].str.contains('PTH'))]

		if filename == 'os_data' or filename == 'ios_data':
			df = df[pd.notna(df['value'])]
			#df = df.head(10)
			df['femur'] = df['value'].apply(get_femur_bmd)
			df[['femur_bmd','femur_t_score','femur_z_score']] = df['femur'].str.split(';',expand=True)
			df.drop(['femur'], axis=1, inplace=True)
			df['lumbar'] = df['value'].apply(get_lumbar_bmd)
			df[['lumbar_bmd','lumbar_t_score','lumbar_z_score']] = df['lumbar'].str.split(';',expand=True)
			df.drop(['lumbar'], axis=1, inplace=True)
			df['radius'] = df['value'].apply(get_radius_bmd)
			df[['radius_bmd','radius_t_score','radius_z_score']] = df['radius'].str.split(';',expand=True)
			df.drop(['radius'], axis=1, inplace=True)
			df.drop(['value'], axis=1, inplace=True)

		df = df.drop_duplicates()
		
		print(df)
		df.to_csv(path+'clean/'+filename+'.csv')

def display_history():
	import matplotlib.pyplot as plt
	import matplotlib.dates as mdate
	df = pd.read_csv(path+'clean/demo_data.csv', index_col=0)
	lab = pd.read_csv(path+'clean/lab_data.csv', index_col=0)
	drug = pd.read_csv(path+'clean/drug_data.csv', index_col=0)
	bmd = pd.read_csv(path+'clean/os_data.csv', index_col=0)
	hn = df['hn'].values.tolist()
	hn = list(dict.fromkeys(hn))

	for i in hn:
		d = df[df['hn']==i]
		d['date'] = pd.to_datetime(d['date'])
		d['year'] = d['date'].dt.year
		txn = d['txn'].values.tolist()
		l = lab[lab['txn'].isin(txn)]
		l['date'] = pd.to_datetime(l['date'])
		l['year'] = l['date'].dt.year
		l = l[pd.notna(l['value'])]
		dr = drug[drug['txn'].isin(txn)]
		dr['date'] = pd.to_datetime(dr['date'])
		dr['year'] = dr['date'].dt.year
		dr['value'] = dr['name'].str.len()
		b = bmd[bmd['txn'].isin(txn)]
		b['date'] = pd.to_datetime(b['date'])
		b['year'] = b['date'].dt.year
		'''
		if len(l) > 0:
			print(d)
			print(l)
			break
		'''
		if len(l) > 0:
			#fig = plt.figure()
			#fig.set_size_inches(8.5, 5.5, forward=True)
			#ax = fig.add_subplot(3, 1, 1, facecolor='#E6E6E6')
			fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex='all')
			fig.set_size_inches(12.5, 8.5, forward=True)
			f1 = 'date'
			f2 = 'demographic'
			ax1.set_xlabel('age')
			ax2.set_xlabel('weight')
			ax3.set_xlabel('lab')
			ax4.set_xlabel('drug')
			ax5.set_xlabel('bmd')
			d1 = d[['year','age']]
			d1.drop_duplicates(inplace=True)
			d2 = d[['year','weight']]
			d2.drop_duplicates(inplace=True)
			ax1.scatter(d1['year'], d1['age'], alpha=0.5, c='green', label='age')
			ax2.scatter(d2[d2['weight']>0]['year'], d2[d2['weight']>0]['weight'], alpha=0.5, c='red', label='weight')
			ax3.scatter(l['year'], l['value'], alpha=0.5, c='blue', label='lab')
			ax4.scatter(dr['year'], dr['value'], alpha=0.5, c='purple', label='drug')
			ax5.scatter(b['year'], b['lumbar_bmd'], alpha=0.5, c='black', label='lumbar bmd')
			ax5.scatter(b['year'], b['lumbar_z_score'], alpha=0.5, c='gray', label='lumbar bmd (z-score)')
			ax5.scatter(b['year'], b['lumbar_t_score'], alpha=0.5, c='gray', label='lumbar bmd (t-score)')
			#ax.scatter(l['date'], l['value'], alpha=0.5, c='blue', edgecolors='none', s=30, label='lab')
			#ax1.xaxis.set_major_locator(years)
			#ax1.xaxis.set_major_formatter(years_fmt)
			#plt.text(1,1,'case '+d['sex'].iloc[0])
			#plt.title('Demographic data')
			#plt.legend(loc=1)
			ax1.legend(loc=1)
			ax2.legend(loc=1)
			ax3.legend(loc=1)
			ax4.legend(loc=1)
			ax5.legend(loc=1)
			
			for index, row in l.iterrows():
				ax3.text(row['year']+0.1,row['value'],row['name'])
			for index, row in dr.iterrows():
				ax4.text(row['year']+0.1,row['value'],row['name'],fontsize=6)
			plt.show()
			#break



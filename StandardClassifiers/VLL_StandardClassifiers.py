import numpy as np
from scipy.io import arff as arff2
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing 
from collections import defaultdict
from sklearn import base
d = defaultdict(preprocessing.LabelEncoder)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.dummy import *
from sklearn.linear_model import LogisticRegression

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GroupKFold


import pandas
import csv

from sklearn.model_selection import cross_val_score

metrics = [ f1_score, precision_score, recall_score, roc_auc_score]
featureSets = []

def writeListToFile(oStream, l):
	with open(oStream, 'a') as out:
			out.write("\t".join(l))
			out.write("\n")

def trainClassifier(features, labels, testF, testL, classifier,prior, name="NoName", kF = "NA"):
	print "Training Classifier"
	clf = base.clone(classifier)
	clf.fit(features,labels)
	acc = clf.score(testF,testL)
	p = clf.predict(testF)
	
	print "Evaluating Classifier"

	#Output Information
	output = []
	output.append(name)	
	output.append(str(clf.__class__.__name__))
	output.append(str(len(list(features.columns.values))))
	output.append(str(len(features.index)+len(testL.index)))
	output.append(str(prior))
	output.append(str(kF))

	#Add Metircs to output
	val = r2_score(testL, p)
	output.append(str(val))

	val = spearmanr(testL, p)
	output.append(str(val[0]))

	val = pearsonr(testL, p)
	output.append(str(val[0]))

	return output

def runClassifier_k(name, df, oStream, classifier, feats, label, k=2, id_col = []):
		prior = df[label].mean()
		
		for iter_num, fold_indices in enumerate(GroupKFold(k).split(X=df, groups=df[id_col])):
			train_indices = fold_indices[0]
			test_indices = fold_indices[1]

			train_inst = df.iloc[train_indices].copy()

			f = feats[1]

			features = train_inst[f]
			labels = train_inst[label]


			

			test_inst  = df.iloc[test_indices].copy()
			testF = test_inst[f]
			testL = test_inst[label]

			output = trainClassifier(features,labels,testF,testL, classifier, prior,name,iter_num)

			writeListToFile(oStream, output)


def runExperiment_survey(fsPass, classLabel):
	for featureSet in fsPass:
		for f in files:
			for meth in methods:
				data = pandas.read_csv(datadir+f, sep=",")

				sub = data
				#sub=data
				#print sub[moderator]
				#print sub['survey_question'].dtype
				#print sub['survey_question'].unique()
				sub = data[data['survey_question'] == classLabel]

				#sub = sub[sub[classLabel] != 'nan']
				
				sub['survey_answer'] = sub['survey_answer'].astype(int)
				

				data = sub
				#data = data.sample(frac=0.33)
				#data = sub.apply(lambda x: d[x.name].fit_transform(x))
				
				runClassifier_k(classLabel+"_"+featureSet[0], data, outputDir+output, meth[0], featureSet, "survey_answer", 10, GroupSplit)
	

#### Global Variable Setup
	
output = "VLL_T1.txt"

files = [ "n_5_FullYear_v4.csv"]

methods = []

Oversample = [0]

datadir = ""

outputDir =  "Output\\"

GroupSplit = "student_id"


###### Classifiers to run ###########

#methods.append((MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 15), random_state=1), "MLP"))
#methods.append((RandomForestClassifier(n_estimators = 100), "RandForest"))
#methods.append((GaussianNB(), "GaussianNB"))
#methods.append((MultinomialNB(), "MultinomialNB"))
#methods.append((DecisionTreeClassifier(), "DecisionTreeClassifier"))

#methods.append((DummyClassifier(strategy='stratified'), "Dummy"))
methods.append((linear_model.BayesianRidge(),"BaysianRidge"))
methods.append((linear_model.ElasticNet(),"ElasticNet"))

#methods.append((GradientBoostingClassifier(loss='exponential'),"GradientBoosting_EXP"))
#methods.append((GradientBoostingClassifier(),"GradientBoosting_dev"))
#
methods.append((linear_model.LinearRegression(), "LinearRegression"))


#methods.append((RandomForestRegressor(n_estimators= 50), "RandomForestRegressor50"))
methods.append((RandomForestRegressor(n_estimators= 100), "RandomForestRegressor100"))

labels = ["Anxiety","Arousal","Boredom","Confusion","Contentment","Curiosity","Disappointment","Engagement","Frustration","Happiness","Hopefulness","Interest","Mind Wandering","Pleasantness","Pride","Relief","Sadness","Surprise"]

vll5_real = ("5SecFull", ["bio_video_watch","karma_awarded","leaderboard_load","ortan_load","ortan_video_watch","personal_profile_picture","tys_answer","tys_finish","tys_load","tys_previous","tys_review_correct_question","tys_review_incorrect_question","tys_review_solution_video","tys_review_topic_video","tys_unload","video_caption","video_completed","video_pause","video_play","video_seek","video_watch","wall_load_more","wall_make_post","wall_page_load","wall_refresh","wall_search"])

setToRun = [vll5_real]


for s in setToRun:
	for l in labels:
		featureSets = [] 
		featureSets.append(s)
		
		
		runExperiment_survey(featureSets, l)


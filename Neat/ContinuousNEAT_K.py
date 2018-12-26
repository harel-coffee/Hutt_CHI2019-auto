import numpy as np
from scipy.io import arff as arff2
from scipy.stats import spearmanr
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
from sklearn.model_selection import *

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




#import arff
import pandas
import csv

import time

import neat
import math
import sys
from datetime import datetime

from sklearn.model_selection import cross_val_score

metrics = [ f1_score, precision_score, recall_score, roc_auc_score]
featureSets = []

#Global Variables used in Model Training
inputs = []
outputs = []

testIn = []
testOut = []

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        #genome.fitness = -4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        p = []
        ##print inputs
        #exit()
        for xi, xo in zip(inputs, outputs):
            #print(xi)
            pred = net.activate(xi)
            p.append(pred)
        ##print p
        f = spearmanr(outputs, p)[0]
        if math.isnan(f):
        	f = -1
        genome.fitness = f
        ##print genome.fitness
#content ='Mex4-2.arff'
def run(config_file, outcome = "Neat-Exp", k = "0"):
    # Load configuration.
    ##print neat.__version__
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=outputDir+ "/" +outcome +"_" + str(k) +  '-neat-checkpoint-'))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    stats.save_genome_fitness(delimiter=',', filename=outputDir+ "/" + outcome+"_" + str(k) +'_fitness_history.csv', with_cross_validation=False)

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    preds =[]
    for xi, xo in zip(testIn, testOut):
        output = winner_net.activate(xi)[0]
        preds.append(output)

    #print (testOut)
    #print("******PREDS*****")
    #print(preds)
    f = spearmanr(testOut, preds)[0]
    #print (spearmanr(testOut,preds))

    out=[]
    out.append(outcome)
    #out.append(str(winner))
    out.append(str(f))
    with open(outputDir+ "/" +"TestSet.csv", 'a') as o:
        o.write(",".join(out))
        o.write("\n")



def neatMain(df, oStream, folds, feats, level, mod, outcome, label="survey_answer"):
    #print("In Neat method ")
    print("Begin NEAT Preamble")
    print ("*****************************************************************************************")
    print (mod)
    print ("*****************************************************************************************")
    #print (neat.__version__)
    global inputs
    global outputs
    global testIn
    global testOut
    kf = KFold(n_splits=10, shuffle=True)
    pstore = []
    s=feats[0]
    for iter_num, fold_indices in enumerate(GroupKFold(folds).split(X=df, groups=df[id_col])):
        train_indices = fold_indices[0]
        test_indices = fold_indices[1]

        train_inst = df.iloc[train_indices].copy()

        f = s[1]


        features = train_inst[f].values

        labels = train_inst[label].values



        inputs = features
        outputs = labels

        test_inst  = df.iloc[test_indices].copy()
        testIn = test_inst[f].values
        testOut = test_inst[label].values


        run("NEAT_VLL_2_Full", mod, iter_num)

def runExperiment_survey_NEAT(fsPass, classLabel, f):
    for f in files:
        data = pandas.read_csv(datadir+f, sep=",")

        sub = data
        #sub=data
        ##print sub[moderator]
        #print(sub['survey_question'].dtype)
        print(sub['survey_question'].unique())
        try:
            sub = data[data['survey_question'] == classLabel].copy()
        except:
            print("Invalid Command Line Arguments, Affective state not found")
            print("Available affective states in current file are:")
            print(sub['survey_question'].unique())
            exit()

        #sub = sub[sub[classLabel] != 'nan']

        sub['survey_answer'] = sub['survey_answer'].astype(int)
        data = sub


        #data = data.sample(frac=0.33)
        data = sub.apply(lambda x: d[x.name].fit_transform(x))
        #runClassifier(meth[1],data, outputDir+output, meth[0], 4, over, dicts, fsPass, 0,  "AllInst", f, "survey_answer", classLabel, f)
        print (list(data.columns.values))
        neatMain(data, outputDir+output, 10, fsPass, 0, classLabel, f[:5])





output = "VLL_Continuous_FullYear_Round2_3Sub_3Window.txt"

#files = [  "RESPONDED_Alg1_1min_SEQUENCE_features By (student_id, survey_id, survey_question, survey_answer).csv", "RESPONDED_Alg1_3min_SEQUENCE_features 2 By (student_id, survey_id, survey_answer, survey_question).csv"]
#files = [  "6YearWithHS.csv"]
files = ["n_5_FullYear_v4.csv"]

id_col = "student_id"
methods = []

Oversample = [0]

datadir = ""

outputDir =  "Output\\"

InstanceFilter = 'FirstEnrolledInstGradRate4Yr'

#methods.append((MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 15), random_state=1), "MLP"))
#methods.append((RandomForestClassifier(n_estimators = 100), "RandForest"))
methods.append((GaussianNB(), "GaussianNB"))

vll5_real = ("5SecFull", ["bio_video_watch","karma_awarded","leaderboard_load","ortan_load","ortan_video_watch","personal_profile_picture","tys_answer","tys_finish","tys_load","tys_previous","tys_review_correct_question","tys_review_incorrect_question","tys_review_solution_video","tys_review_topic_video","tys_unload","video_caption","video_completed","video_pause","video_play","video_seek","video_watch","wall_load_more","wall_make_post","wall_page_load","wall_refresh","wall_search"])

setToRun = [vll5_real]
try:
	l = sys.argv[1]
except:
    print("Invalid Command Line Arguments, please include which survey you wish to build a model for.")
    exit()

for f in files:
	#global outputDir


	outputDir = "Run_k_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f[15:19]

	import os
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
	for s in setToRun:

			featureSets = []
			featureSets.append(s)
			runExperiment_survey_NEAT(featureSets, l, f)

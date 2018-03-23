import os
import numpy as np
import scipy as sp
import sklearn as sl
import sklearn.metrics as met
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ted_talk_sentiment import Sentiment_Comparator, read_bluemix
from TED_data_location import ted_data_path
import ted_talk_cluster_analysis as tca

kwlist = ['beautiful', 'ingenious', 'fascinating',
            'obnoxious', 'confusing', 'funny', 'inspiring',
             'courageous', 'ok', 'persuasive', 'longwinded', 
             'informative', 'jaw-dropping', 'unconvincing','Totalviews']

# def traintest_idx(N,testsize=0.3):
#     '''
#     Get the index of training and test split. 
#     N is the length of the dataset (sample size)
#     '''
#     testidx = np.random.rand(int(N*testsize))*N
#     testidx = testidx.astype(int).tolist()
#     trainidx = [i for i in xrange(N) if not i in testidx]
#     return trainidx,testidx

# def discretizeY(Y,col,firstThresh=33.3333,secondThresh=66.6666):
#     '''
#     Discretize and returns and specific column of Y. The strategy is:
#     to keep the data with score <=33rd percentile be the "low" group,
#     score >=66th percentile be the "high" group, and the middle be the
#     "medium" group.
#     '''
#     y = Y[:,col]
#     if kwlist[col] == 'Totalviews':
#         y=np.log(y)
#     lowthresh = sp.percentile(y,firstThresh)
#     hithresh = sp.percentile(y,secondThresh)
#     y[y<=lowthresh] = -1    # Low group
#     y[y>=hithresh] = 1      # High group
#     y[(y>lowthresh)*(y<hithresh)] = 0   # Medium group
#     return y

# def binarize(X,y):
#     '''
#     Keeps only the good and bad parts in the data. Drops the medium part.
#     But if there is only two part, then just repair the labels
#     '''
#     unqy = np.unique(y)
#     if len(unqy)==3:
#         idxmed = y!=0
#         return X[idxmed,:],y[idxmed]
#     elif len(unqy)==2 and 0 in unqy and -1 in unqy:
#         y[y==0]=1
#         return X,y
#     else:
#         raise IOError
    

def classifier_eval(clf_trained,X_test,y_test,use_proba=True,
        ROCTitle=None,outfilename='TED_stats/classifier_results.png'):
    y_pred = clf_trained.predict(X_test)
    print sl.metrics.classification_report(y_test,y_pred)
    print 'Accuracy:',sl.metrics.accuracy_score(y_test,y_pred)
    if use_proba:
        try:
            # trying to get the confidence scores
            y_score = clf_trained.decision_function(X_test)
        except AttributeError:
            print 'model does not have any method named decision function'
            print 'Trying predict_proba:'
            try:
                y_score = clf_trained.predict_proba(X_test)
            except:
                raise
        auc = met.roc_auc_score(y_test,y_score)
        print 'AUC:',auc
        fpr,tpr,_ = sl.metrics.roc_curve(y_test,y_score,pos_label=1)        
        plt.figure(0)
        plt.clf()
        plt.plot(fpr,tpr,color='darkorange',label='ROC Curve (AUC={0:0.2f})'.\
            format(auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if ROCTitle:
            plt.title(ROCTitle)
        plt.legend()
        if not outfilename:
            plt.show()
        else:
            outfilename = os.path.join(ted_data_path,outfilename)
            split_fn = os.path.split(outfilename)
            plt.savefig(os.path.join(split_fn[0],ROCTitle+'_'+split_fn[1]))
            plt.close()
        
def regressor_eval(regressor_trained,X_test,y_test):
    y_pred = regressor_trained.predict(X_test)
    print 'Corr.Coeff:{0:2.2f} '.format(np.corrcoef(y_test,y_pred)[0,1]),
    print 'MSE:{0:2.2f} '.format(met.mean_squared_error(y_test,y_pred)),
    print 'MAE:{0:2.2f} '.format(met.mean_absolute_error(y_test,y_pred)),
    print 'MedAE:{0:2.2f} '.format(met.median_absolute_error(y_test,y_pred)),
    print 'EVSc.:{0:2.2f} '.format(met.explained_variance_score(y_test,y_pred)),
    print 'R2S.:{0:2.2f} '.format(met.r2_score(y_test,y_pred)),
    print 'Smpl:',len(y_test)

def train_with_CV(X,y,predictor,cvparams,
        score_func=met.roc_auc_score,Nfold=3,nb_iter=10,
        showCV_report=False,use_proba=True,datname=''):
    '''
    Trains the estimator with N fold cross validation. The number of fold
    is given by the parameter Nfold. cvparams is a dictionary specifying
    the hyperparameters of the classifier that needs to be tuned. Scorefunc
    is the metric to evaluate the classifier. 
    If the number of unique y values are <=3, then the predictor is assumed
    to be a classifier. Otherwise, it is assumed to be a regressor. The
    assumption of classifier/regresssor is used when evaluating the predictor.
    For a classifier, the default scorer is roc_auc_score, for regressor,
    default scorer is r2_score
    '''    
    if len(np.unique(y))<=3:
        predictor_type = 'classifier'
    else:
        predictor_type = 'regressor'
    # If classifier, use the given scorefunction. If regressor, and the
    # given scorefunction is the default one, use the default regressor score.
    # Otherwise, just use the given scorefunction.
    if predictor_type == 'classifier' or (predictor_type == 'regressor' and \
        not score_func == met.roc_auc_score):
        scorer = sl.metrics.make_scorer(score_func)
    else:
        scorer = sl.metrics.make_scorer(met.r2_score)
    # Perform cross-validation
    randcv = sl.model_selection.RandomizedSearchCV(predictor,cvparams,
                n_iter=nb_iter,scoring=scorer,cv=Nfold)
    
    randcv.fit(X,y)
    y_pred = randcv.best_estimator_.predict(X)
    print 'Report on Training Data'
    print '-----------------------'
    print 'Best Score:',randcv.best_score_
    # Evaluate the predictor
    if predictor_type=='classifier':
        classifier_eval(randcv.best_estimator_,X,y,use_proba,
            'ROC on Training Data '+datname)
    else:
        regressor_eval(randcv.best_estimator_,X,y)
    if showCV_report:
        print 'CV Results:'
        print randcv.cv_results_
    return randcv.best_estimator_,randcv.best_score_
    

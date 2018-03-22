import ted_talk_sentiment as ts
import ted_talk_data_feeder as ttdf
import ted_talk_cluster_analysis as tca
import ted_talk_prediction as tp
from ted_talk_statistic import plot_statistics
from ted_talk_statistic_correlation import plot_correlation
from TED_data_location import ted_data_path

from list_of_talks import allrating_samples, all_valid_talks, hi_lo_files

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import sklearn as sl
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# This python file enlists many experiments we have done.
# It can also be used as sample usage of the code repository such as
# the sentiment_comparator class.
# Bluemix sentiments:
# ==================
# 0: anger 
# 1: disgust 
# 2: fear 
# 3: joy 
# 4: sadness 
# 5: analytical 
# 6: confident 
# 7: tentative 
# 8: openness_big5 
# 9: conscientiousness_big5 
# 10: extraversion_big5 
# 11: agreeableness_big5 
# 12: emotional_range_big5


# Get some sample datapoints just for testing
comparator = ts.Sentiment_Comparator(hi_lo_files)     

def bluemix_plot1(outfile = 'bm_plot1.png'):
    '''
    This function plots the progression of average <b>emotion scores</b>
    for 30 highest viewed ted talks and 30 lowest viewed ted talks.
    If you want to save the plots in a file, set the outfilename argument.
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/' + outfile)
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[0,1,2,3,4],   # only emotion scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center',
        outfilename=outfilename
        )
    print 'File saved in:',outfilename

def bluemix_plot2(outfilename='bm_plot2.png'):
    '''
    This function plots the progression of average Language scores for 30 
    highest viewed ted talks and 30 lowest viewed ted talks. If you want
    to save the plots in a file, set the outfilename argument.
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/' + outfilename)
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[5,6,7],   # only Language scores
        styles=['r.--','r-','r--',
                'b.--','b-','b--'],  # appropriate line style
        legend_location='lower center',
        outfilename=outfilename
        )
    print 'File saved in:',outfilename

def bluemix_plot3(outfilename='bm_plot3.png'):
    '''
    This function plots the progression of average Social scores for 30 
    highest viewed ted talks and 30 lowest viewed ted talks. If you want
    to save the plots in a file, set the outfilename argument.
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/' + outfilename)
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[8,9,10,11,12],   # only big5 scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center',
        outfilename=outfilename
        )
    print 'File saved in:',outfilename


def bluemix_plot4(outprefix='plots_',ext='.png'):
    '''
    This function plots the progression of all the scores one by one.
    The average was calculated for 30 highest viewed ted talks and 30
    lowest viewed ted talks. By default, the plots are saved with their
    unique names inside the directory specified by outprefix argument.
    If you want to see the plots in window, set outprefix to None
    '''
    outpath = os.path.join(ted_data_path,'TED_stats/')
    avg_ = comparator.calc_group_mean()
    for i in range(13):
        if outprefix:
            outfname = os.path.join(outpath, outprefix + \
                comparator.column_names[i]+ext)
        else:
            outfname = None
        # Plot Group Average
        ts.draw_group_mean_sentiments(avg_, # the average of groups
            comparator.column_names,        # name of the columns
            selected_columns=[i],   # only emotion scores
            styles=['r-',
                    'b-'],  # appropriate line style
            legend_location='lower center',
            outfilename=outfname)
        print 'File saved in:',outfname


def bluemix_plot5(outfilename='hivi_lovi.png'):
    '''
    This function plots the time averages for the 30 highest viewed
    and 30 lowest viewed ted talks. In addition, it performs T-tests
    among the hi-view and lo-view groups. By default, the output is saved
    in the './plots/hivi_lovi.png' file. But if you want to see it
    on an interactive window, just set outfilename=None
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/'+outfilename)
    avg_,p = comparator.calc_time_mean()
    ts.draw_time_mean_sentiments(avg_, # time averages
        comparator.column_names,       # name of the columns
        p,                              # p values
        outfilename=outfilename
        )
    print 'File saved in:',outfilename

def single_plot(talkid = 2774, selected_scores = [1,3,12],
    draw_full_y=False, outfilename='<talkid>.png'):
    '''
    Plots the score progression for a single talk.
    Note that this function does not plot the raw score.
    It smoothens the raw score value, cuts the boundary distortions
    (due to smoothing) and interpolates from 0 to 100 before showing
    the plots.
    The selected_scores argument defines which scores to show. Showing
    too many scores at once will make the plot busy.
    If draw_full_y is set True, the plots are drawn over a y-axis ranging
    from 0 to 1.
    If outfilename is set to a filename, the plot is saved to that file.
    The indices of bluemix scores are as follows (needed in the selected
    scores argument):
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/'+\
        outfilename.replace('<talkid>',str(talkid)))
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk)
    ts.draw_single_sentiment(\
        comp.sentiments_interp[talkid], # plot the interpolated sentiment
        comp.column_names,              # Name of the columns
        selected_scores,                # Show only Disgust, Joy and Emotional
        full_y=draw_full_y,
        outfilename = outfilename 
        )
    print 'output saved at:',outfilename

def single_plot_raw(talkid, selected_scores=[3,4],
    draw_full_y=False, outfilename='<talkid>.png'):
    '''
    Plots the <b>Raw</b> score progression for a single talk.
    The selected_scores argument defines which scores to show. Showing
    too many scores at once will make the plot busy.
    If draw_full_y is set True, the plots are drawn over a y-axis ranging
    from 0 to 1.
    If outfilename is set to a filename, the plot is saved to that file.
    The indices of bluemix scores are as follows (needed in the 
    selected_scores argument):
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/'+\
        outfilename.replace('<talkid>',str(talkid)))
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk,process=False)
    comp.extract_raw_sentiment()
    ts.draw_single_sentiment(\
        comp.raw_sentiments[talkid], # plot the interpolated sentiment
        comp.column_names,              # Name of the columns
        selected_scores,                # Show only Disgust, Joy and Emotional
        full_y=draw_full_y,
        outfilename = outfilename 
        )
    print 'output saved at:',outfilename

def single_plot_smoothed(talkid=2774,selected_scores=[3,4],
    draw_full_y=False,outfilename='<talkid>.png'):
    '''
    Plots the Smoothed (but not interpolated) score progression for a
    single talk. The selected_scores argument defines which scores to 
    show. Showing too many scores at once will make the plot busy.
    If draw_full_y is set True, the plots are drawn over a y-axis ranging
    from 0 to 1.
    If outfilename is set to a filename, the plot is saved to that file.
    The indices of bluemix scores are as follows (needed in the 
    selected_scores argument):
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5
    '''
    if outfilename:
        outfilename = os.path.join(ted_data_path,'TED_stats/'+\
        outfilename.replace('<talkid>',str(talkid)))
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk)
    ts.draw_single_sentiment(\
        comp.raw_sentiments[talkid], # plot the interpolated sentiment
        comp.column_names,              # Name of the columns
        selected_scores,                # Show only Disgust, Joy and Emotional
        full_y=draw_full_y,
        outfilename = outfilename 
        )
    print 'output saved at:',outfilename
    
def see_sentences_percent(talkid,start=0,end=100,selected_scores=None):
    '''
    Prints the sentences of a talk from a start percent to end percent.
    Notice that the start and end indices are numbered in terms of
    percentages of the the talk. The percentages are automatically
    converted back to the raw indices of each sentence.
    This function also shows the scores for each sentence. Use the
    selected_scores argument to specify which scores you want to see.
    By default, it is set to None, which means to show all the scores
    for each sentence.
    '''
    # Display sample sentences
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk)
    comp.display_sentences(talkid, # Talk ID
        start, # Start percent
        end,  # End percent
        selected_columns = selected_scores
        )

def time_avg_hi_lo_ratings():
    '''
    Experiment on High/Low ratings
    '''
    avg_saved = np.array([])
    i = 0
    for a_grp_dict in allrating_samples:
        i = i+1
        allkeys = sorted(a_grp_dict.keys())
        titl = allkeys[0]+' vs. '+allkeys[1]
        print titl
        compar = ts.Sentiment_Comparator(
            a_grp_dict     # Compare between hi/lo viewcount files
            )
        avg_,p = compar.calc_time_mean()
        avg_saved = np.append(avg_saved, avg_)
 
    return avg_saved

def time_avg_hi_lo_ratings_original(outfilename='time_<title>.png'):
    '''
    Experiment on the time average of (30) Highly rated talks and 
    low rated talks. 
    Besides calculating the time average, it also calculates
    the p-values for t-tests showing if there is any difference in 
    the average scores.
    '''
    avg_saved = np.array([])
    for a_grp_dict in allrating_samples:
        allkeys = sorted(a_grp_dict.keys())
        titl = allkeys[0]+' vs. '+allkeys[1]
        print titl
        compar = ts.Sentiment_Comparator(
            a_grp_dict     # Compare between hi/lo viewcount files
            )
        avg_,p = compar.calc_time_mean()
        filename = os.path.join(ted_data_path,'TED_stats/'+\
            outfilename.replace('<title>',titl.replace(' ','_')))
        ts.draw_time_mean_sentiments(avg_, # time averages
           comparator.column_names,       # name of the columns
           p,                             # p values                      
           outfilename=filename
        )
        print 'Saved as:',filename

def grp_avg_hilo_ratings(score_list=[[0,1,2,3,4],[5,6,7],[8,9,10,11,12]],
    outfilename='grp_<title>.png'):
    '''
    Experiment on the (ensemble) average of scores for 30 Highly rated
    talks and 30 low rated talks. 
    For every rating, it attempts to show the averages of various scores.
    The score_list is a list of list indicating which scores would be
    grouped together in one window. By default, the emotional, language,
    and personality scores are grouped together. The indices of the scores
    are given below:
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5

    The plots are saved in ./plots/ directory.
    '''
    for a_grpdict in allrating_samples:
        allkeys = sorted(a_grpdict.keys())
        titl = allkeys[0]+' vs. '+allkeys[1]+' group average'
        print titl
        filename = os.path.join(ted_data_path,'TED_stats/'+\
            outfilename.replace('<title>',titl.replace(' ','_')))
        compar = ts.Sentiment_Comparator(
            a_grpdict     # Compare between hi/lo viewcount files
            )
        grp_avg = compar.calc_group_mean()
        for i in score_list:
            if len(i)==1:
                styles = ['r-','b-']
            elif len(i)==2:
                styles = ['r^-','r--',
                 'b^-','b--']                
            elif len(i)==3:
                styles = ['r^-','r--','r-',
                 'b^-','b--','b-']
            else:
                styles = ['r^-','r--','r-','r.-','ro-',
                 'b^-','b--','b-','b.-','bo-']

            ts.draw_group_mean_sentiments(grp_avg,
                compar.column_names,
                i,
                styles,
                outfilename=filename)
            print 'Saved as:',filename

def draw_global_means(comp,ext='.png'):
    '''
    Experiment on the global average of sentiment progressions in
    ALL* tedtalks
    * = all means the 2007 valid ones.
    Use the following commands to generate comp where ts is the
    ted_talk_sentiment.py module
    comp = ts.Sentiment_Comparator({'all':all_valid_talks})
    '''
    avg = comp.calc_group_mean()['all']
    plt.figure(figsize=(6.5,6))
    grpnames = ['Emotion Scores', 'Language Scores', 'Personality Scores']
    for g,agroup in enumerate([[0,1,2,3,4],[5,6,7],[8,9,10,11,12]]):
        groupvals = np.array([avg[:,acol] for acol in agroup]).T
        import re
        colnames = [re.sub(\
            'emotion_tone_|language_tone_|social_tone_|_big5',\
            '',comp.column_names[acol]) for acol in agroup]

        plt.subplot(3,1,g+1)
        plt.plot(groupvals)
        plt.xlabel('Percent of Talk')
        plt.ylabel('Value')
        plt.ylim([[0,0.6],[0,0.5],[0.2,0.6]][g])
        #plt.subplots_adjust(bottom=0.05, right=0.99, left=0.05, top=0.85)
        #plt.legend(colnames,bbox_to_anchor=(0., 1.05, 1., 0), loc=3,\
        #   ncol=2, mode="expand", borderaxespad=0.)
        plt.legend(colnames,ncol=[5,3,3][g],loc=['upper left',\
            'upper left','lower left'][g])
        plt.title(['Emotion Scores','Language Scores','Personality Scores'][g])
        plt.tight_layout()
    filename = os.path.join(ted_data_path,'TED_stats/global_scores'+ext)
    plt.savefig(filename)
    print 'saved as:',filename


def clusters_pretty_draw(X,comp,outfilename='TED_stats/draw_clusters_pretty.png'):
    '''
    Draws the top 20 talks most similar to the cluster means
    and name five of them

    Note: before you call this function, you should get the arguments
    (X and comp) using the following command: 
    X,comp = tca.load_all_scores()
    tca is the ted_talk_cluster_analysis module
    load_all_scores is a slow function      
    '''
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(eps = 6.5, min_samples = 5)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    avg_dict=tca.clust_separate_stand(X,km,comp,\
        csvcontent,csv_vid_idx)
    outfilename = os.path.join(ted_data_path,outfilename)
    tca.draw_clusters_pretty(avg_dict,comp,csvcontent,csv_vid_idx,
        outfilename=outfilename)
    print 'Group of out files:',outfilename

# def decide_best_cluster_parameters(X,comp,paramlist):
#     '''
#     Given a list of cluster parameters, it attempts to determine the best
#     parameters by sorting them from best to worst. The quality is measured
#     by computing the average (per cluster) distance of the interpolated 
#     scores from the group means.
#     '''
#     quality={}
#     for params in paramlist:
#         N,M,B = X.shape
#         avg_dict = {}
#         clusterer = DBSCAN(**params)
#         # For every bluemix score
#         for s in range(B):
#             clust_dict = clust_onescore_stand(X[:,:,s],clusterer,comp,True)
#             comp.reform_groups(clust_dict)
#             clust_avg = comp.calc_group_mean()
#             for aclust in clust_avg:
#                 clust_avg[aclust] - 


# TODO: Try performing clustering without standardization
def evaluate_clusters_pretty(X,comp,outfilename='TED_stats/eval_pretty.png'):
    '''
    Similar to clusters_pretty_draw, but it also computes box plots of the
    ratings in order to evaluate the quality of the clusters in terms of
    rating separations. 
    It also performs an ANOVA test to check if the clusters have
    any differences in their ratings.
    It also performs the following: (Based on CHI Reviewer's recommendations)
    1. ANOVA with Bonferroni correction
    2. Pairwise multiple t-test with Bonferroni correction
    3. Effectsize and direction of the clusters on the ratings

    Note: before you call this function, you should get the arguments
    (X and comp) using the following command: 
    X,comp = tca.load_all_scores()
    tca is the ted_talk_cluster_analysis module
    load_all_scores is a slow function     
    '''
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(eps=7.75, min_samples = 15)
    # km = SpectralClustering(n_clusters = 7, eigen_solver = 'arpack')
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    outfilename = os.path.join(ted_data_path,outfilename)
    cluster_means = tca.evaluate_clust_separate_stand(X,km,comp,csvcontent,
        csv_vid_idx,outfilename=outfilename)
    cluster_mean_file = os.path.join(ted_data_path,'misc/cluster_means.pkl')
    cp.dump(cluster_means,open(cluster_mean_file,'wb'))
    print 'Group of out files:',outfilename
    print 'Cluster means saved in:',cluster_mean_file
    
def classify_multimodal(classifier='logistic_l1',nb_tr_iter=10):
    '''
    Classify between groups of High ratings and low ratings using
    LinearSVM, SVM_rbf and logistic regression. The classifier
    argument can take these two values.
    This function trains the classifiers and evaluates their performances.

    Use the following command to get the initial arguments:
    scores,Y,_ = tp.loaddata()
    tp = ted_talk_prediction module
    Note: loaddata is a slow function
    '''
    print 'Reading Features ...',
    # Get body lanugage feature
    X,label = ttdf.read_openpose_feat()
    # Add facial features
    X,label = ttdf.concat_features(X,label,*ttdf.read_openface_feat())
    # Add sentiment features
    X,label = ttdf.concat_features(X,label,*ttdf.read_sentiment_feat(X.keys()))
    # Add Prosody features
    # Add Lexical features
    # Add Storytelling (clusters) features
    print 'Complete'

    tridx,tstidx = ttdf.split_train_test(X.keys())
    Y,_,ylabels = ttdf.binarized_ratings()

    for i,kw in enumerate(ylabels):
        print
        print
        print kw
        print '================='
        print 'Predictor:',classifier
        # y = tp.discretizeY(Y,i)
        # X_bin,y_bin = tp.binarize(X,y)
        # m = len(y_bin)
        
        # Split in training and test data
        trainX = [X[atalk] for atalk in tridx]
        trainY = [Y[atalk][i] for atalk in tridx]
        testX = [X[atalk] for atalk in tstidx]
        testY = [Y[atalk][i] for atalk in tstidx]
  
        # Classifier selection
        if classifier == 'LinearSVM':
            clf = sl.svm.LinearSVC()
            # Train with training data
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,\
                    {'C':sp.stats.expon(scale=5.)},nb_iter=nb_tr_iter,\
                    datname = kw+'_LibSVM')
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'            
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=kw)
        elif classifier == 'SVM_rbf':
            clf = sl.svm.SVC()
            # Train with training data
            try:
                clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=25),
                    'gamma':sp.stats.expon(scale=0.05)},
                    nb_iter=nb_tr_iter,datname=kw)
                print 'Number of SV:',clf_trained.n_support_
            except ImportError:
                raise
            except:
                print 'Data is badly scaled for',kw
                print 'skiping'
                continue
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'                 
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of SVM_RBF on Test Data for '+kw)
        elif classifier == 'logistic_regression':
            clf = sl.linear_model.LogisticRegression()
            # Train with training data
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=1)},
                    nb_iter=nb_tr_iter,datname=kw)
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'                 
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of SVM_RBF on Test Data for '+kw)
        elif classifier == 'logistic_l1':
            clf = sl.linear_model.LogisticRegression(penalty='l1')
            # Train with training data
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=1)},
                    nb_iter=nb_tr_iter,datname=kw)
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'                 
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of SVM_RBF on Test Data for '+kw)

####### Methods below this line are not ready for new code structure #############

def classify_old(scores,Y,classifier='LinearSVM',nb_tr_iter=10):
    '''
    Classify between groups of High ratings and low ratings using
    Two different types of SVM LinearSVM or SVM_rbf. The classifier
    argument can take these two values.
    This function trains the classifiers and evaluates their performances.

    Use the following command to get the initial arguments:
    scores,Y,_ = tp.loaddata()
    tp = ted_talk_prediction module
    Note: loaddata is a slow function
    '''
    X,nkw = tp.feat_sumstat(scores)
    for i,kw in enumerate(tp.kwlist):
        print
        print
        print kw
        print '================='
        print 'Predictor:',classifier
        y = tp.discretizeY(Y,i)
        X_bin,y_bin = tp.binarize(X,y)
        m = len(y_bin)
        
        # Split in training and test data
        tridx,tstidx = tp.traintest_idx(len(y_bin))
        trainX,trainY = X_bin[tridx,:],y_bin[tridx]
        testX,testY = X_bin[tstidx,:],y_bin[tstidx]
  
        # Classifier selection
        if classifier == 'LinearSVM':
            clf = sl.svm.LinearSVC()
            # Train with training data
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,\
                    {'C':sp.stats.expon(scale=5.)},nb_iter=nb_tr_iter,\
                    datname = kw+'_LibSVM')
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'            
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of LinearSVM on Test Data for '+kw)
        elif classifier == 'SVM_rbf':
            clf = sl.svm.SVC()
            # Train with training data
            try:
                clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=25),
                    'gamma':sp.stats.expon(scale=0.05)},
                    nb_iter=nb_tr_iter,datname=kw)
                print 'Number of SV:',clf_trained.n_support_
            except ImportError:
                raise
            except:
                print 'Data is badly scaled for',kw
                print 'skiping'
                continue
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'                 
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of SVM_RBF on Test Data for '+kw)
        elif classifier == 'logit':
            clf = sl.linear_model.LogisticRegression()
            # Train with training data
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=1)},
                    nb_iter=nb_tr_iter,datname=kw)
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'                 
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of SVM_RBF on Test Data for '+kw)


def regress_ratings(scores,Y,regressor='SVR',cv_score=sl.metrics.r2_score):
    '''
    Try to predict the ratings using regression methods. Besides training
    the regressors, it also evaluates them.

    Use the following command to get the initial arguments:
    scores,Y,_ = tp.loaddata()
    tp = ted_talk_prediction module
    Note: loaddata is a slow function
    '''    
    X,nkw = tp.feat_sumstat(scores)
    for i,kw in enumerate(tp.kwlist):
        print
        print
        print kw
        print '================='
        print 'Predictor:',regressor
        y = Y[:,i]
        if kw == 'Totalviews':
            y=np.log(y)
        tridx,tstidx = tp.traintest_idx(len(y))
        trainX,trainY = X[tridx,:],y[tridx]
        testX,testY = X[tstidx,:],y[tstidx]

        # Predictor Selection
        if regressor=='ridge':
            # Train on training data
            rgrs = sl.linear_model.Ridge()
            rgrs_trained,score = tp.train_with_CV(trainX,trainY,
                rgrs,{'alpha':sp.stats.expon(scale=1.)},
                score_func=cv_score)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)
        elif regressor == 'SVR':
            # Train on training data
            rgrs = sl.svm.LinearSVR(loss='squared_epsilon_insensitive',
                dual=False,epsilon=0.001)
            rgrs_trained,score = tp.train_with_CV(trainX,trainY,
                rgrs,{'C':sp.stats.expon(scale=10)},
                score_func=cv_score)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)
        elif regressor == 'gp':
            # Train on training data
            rgrs = sl.gaussian_process.GaussianProcessRegressor()
            rgrs.fit(trainX,trainY)
            # Evaluate with test data
            print 'Report on Training Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs,testX,testY)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs,testX,testY)
        elif regressor == 'lasso':
            # Train on training data
            rgrs = sl.linear_model.Lasso()
            # Evaluate with test data
            print 'Report on Training Data:'
            print '-----------------------'             
            # Evaluate with training data
            rgrs_trained,score = tp.train_with_CV(trainX,trainY,
                rgrs,{'alpha':sp.stats.expon(scale=0.1)},score_func=cv_score)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)

if __name__=='__main__':
    infolder = './talks/'
    outfolder = './TED_stats/'
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    print '============================================'
    print '============= Ignore Warnings =============='
    print 'Note: The results change at each run due to '
    print 'randomness involved in the predictors       '
    print '============================================'
    print '###### Calculcating dataset statistics #####'
    plot_statistics(infolder,outfolder)
    print '###### Check results in TED_stats folder ###'
    print '##############################################'
    print 'Calculcating dataset statistics (correlations)'
    plot_correlation(False,infolder,outfolder)
    print 'Check results in TED_stats folder'
    print '##############################################'

    print '############ Loading sentiment data ##########'
    comp = ts.Sentiment_Comparator({'all':all_valid_talks},\
        ts.read_bluemix)
    print '############ Calculating global means ########'
    draw_global_means(comp)
    print '####### Check results in the plots folder#####'

    print '##### Loading data for cluster analysis ######'
    X,comp = tca.load_all_scores()
    print '######## Performing cluster analysis #########'
    evaluate_clusters_pretty(X,comp,outfilename='./plots/')
    print '###### Check results in the plot folder ######'
   
    print '### Loading dataset for classif. and regr. ###'
    scores,Y,_ = tp.loaddata()
    print '######### Experimenting on regression ########'
    print 'try: ridge, SVR, gp, lasso'
    regress_ratings(scores,Y,regressor='SVR',\
        cv_score=sl.metrics.r2_score)
    print '###### Experimenting on classification #######'
    print 'try: LinearSVM, SVM_RBF and logit'
    classify_Good_Bad(scores,Y,classifier='LinearSVM')
    print 'Done!'
    

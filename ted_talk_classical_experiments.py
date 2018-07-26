import os
import time
import numpy as np
import scipy as sp
import sklearn as sl
from sklearn.preprocessing import StandardScaler
import cPickle as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering


import ted_talk_sentiment as ts
import ted_talk_data_feeder as ttdf
import ted_talk_cluster_analysis as tca
import ted_talk_prediction as tp
from ted_talk_statistic import plot_statistics
from ted_talk_statistic_correlation import plot_correlation
from TED_data_location import ted_data_path
from list_of_talks import allrating_samples, all_valid_talks, hi_lo_files



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
# Prepare the data loader
def __loaddata__(indexfile='./index.csv'):
    kwlist = ['beautiful', 'ingenious', 'fascinating',
            'obnoxious', 'confusing', 'funny', 'inspiring',
             'courageous', 'ok', 'persuasive', 'longwinded', 
             'informative', 'jaw-dropping', 'unconvincing','Totalviews']
    csv_,vid = tca.read_index(indexfile)
    dict_input = {'all_talks':all_valid_talks}
    # Load into sentiment comparator for all the pre-comps
    comp = ts.Sentiment_Comparator(dict_input)
    scores=[]
    Y=[]
    for atalk in comp.alltalks:
        scores.append(comp.sentiments_interp[atalk])
        temp = []
        for akw in kwlist:
            if akw == 'Totalviews':
                temp.append(int(csv_[akw][vid[atalk]]))
            else:
                temp.append(float(csv_[akw][vid[atalk]])/\
                    float(csv_['total_count'][vid[atalk]])*100.)
        Y.append(temp)
    return np.array(scores),np.array(Y),kwlist,comp    

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
    in the 'outfilename' file. But if you want to see it
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
    X,_,_,comp = __loaddata__()
    tca is the ted_talk_cluster_analysis module
    __loaddata__ is a slow function      
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
        
def evaluate_clusters_pretty(X,comp,outfilename='TED_stats/eval_pretty.png',
    out_clustermeans = 'misc/cluster_params.pkl',
    dbscan_params={'eps':10, 'min_samples':15}):
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
    X,_,_,comp = __loaddata__()
    tca is the ted_talk_cluster_analysis module
    __loaddata__ is a slow function     
    '''
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(**dbscan_params)
    # km = SpectralClustering(n_clusters = 7, eigen_solver = 'arpack')
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    outfilename = os.path.join(ted_data_path,outfilename)
    cluster_means = tca.evaluate_clust_separate_stand(X,
        km,comp,csvcontent,csv_vid_idx,outfilename=outfilename)
    cluster_mean_file = os.path.join(ted_data_path,out_clustermeans)
    dbscan_params['cluster_means']=cluster_means
    cp.dump(dbscan_params,open(cluster_mean_file,'wb'))
    print 'Group of out files:',outfilename
    print 'Cluster means saved in:',cluster_mean_file
    
def classify_multimodal(classifier='logistic_l1',c_scale = 1.,nb_tr_iter=10,
    modality=['pose','face','trajectory','audio','lexical'],
    scale_rating=True,lowerthresh_Y=50.,upperthresh_Y=50.):
    '''
    Classify between groups of High ratings and low ratings using
    LinearSVM, SVM_rbf and logistic regression. The classifier
    argument can take these two values.
    This function trains the classifiers and evaluates their performances.

    Use the following command to get the initial arguments:
    scores,Y,_,_ = __loaddata__()
    tp = ted_talk_prediction module
    Note: loaddata is a slow function
    '''
    old_time = time.time()
    print 'Reading Features ...'
    # Get body lanugage feature
    X={atalk:[] for atalk in all_valid_talks}
    label=[]
    # Add pose features
    if 'pose' in modality:
        X,label = ttdf.concat_features(X,label,*ttdf.read_openpose_feat())
        print 'Openpose features read'

    # Add facial features
    if 'face' in modality:
        X,label = ttdf.concat_features(X,label,*ttdf.read_openface_feat())
        print 'Openface features read'

    # Add sentiment features
    if 'trajectory' in modality:
        X,label = ttdf.concat_features(X,label,\
            *ttdf.read_sentiment_feat(X.keys()))
        print 'Trajectory features read'

    # Add Prosody features
    if 'audio' in modality:
        X,label = ttdf.concat_features(X,label,*ttdf.read_prosody_feat(X.keys()))
        print 'Prosody features read'
    
    # Add Lexical features
    if 'lexical' in modality:
        X,label = ttdf.concat_features(X,label,*ttdf.read_lexical_feat(X.keys()))
        print 'Lexical features read'
    print 'Complete'

    # Train-Test set preparation
    tridx,tstidx = ttdf.split_train_test(talklist=X.keys())
    trainX = [X[atalk] for atalk in tridx]
    testX = [X[atalk] for atalk in tstidx]
    # Feature normalization
    normalizer = StandardScaler().fit(trainX)
    trainX = normalizer.transform(trainX)
    testX = normalizer.transform(testX)

    Y,ylabels=ttdf.binarized_ratings(firstThresh=lowerthresh_Y,\
        secondThresh=upperthresh_Y,scale_rating=scale_rating)

    allresults = {}
    for i,kw in enumerate(ylabels):
        print
        print
        print kw
        print '================='
        print 'Predictor:',classifier
        
        # Split in training and test data        
        trainY = [Y[atalk][i] for atalk in tridx]
        testY = [Y[atalk][i] for atalk in tstidx]
  
        # Classifier selection
        if classifier == 'LinearSVM':
            clf = sl.svm.LinearSVC()
            # Train with training data and crossvalidation
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,\
                    {'C':sp.stats.expon(scale=c_scale)},nb_iter=nb_tr_iter,\
                    datname = kw+'_LibSVM')
            # Evaluate with test data
            print 'Report on Dev Data'
            print '-----------------------'            
            results = tp.classifier_eval(clf_trained,testX,testY)
        elif classifier == 'SVM_rbf':
            clf = sl.svm.SVC()
            # Train with training data and crossvalidation
            try:
                clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=c_scale),
                    'gamma':sp.stats.expon(scale=0.5)},
                    nb_iter=nb_tr_iter,datname=kw)
                print 'Number of SV:',clf_trained.n_support_
            except ImportError:
                raise
            except:
                print 'Data is badly scaled for',kw
                print 'skiping'
                continue
            # Evaluate with test data
            print 'Report on Dev Data'
            print '-----------------------'                 
            # Evaluate with test data
            results = tp.classifier_eval(clf_trained,testX,testY)
        elif classifier == 'logistic_regression':
            clf = sl.linear_model.LogisticRegression(penalty='l2')
            # Train with training data and crossvalidation
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=c_scale)},
                    nb_iter=nb_tr_iter,datname=kw)
            # Evaluate with test data
            print 'Report on Dev Data'
            print '-----------------------'                 
            # Evaluate with test data
            results = tp.classifier_eval(clf_trained,testX,testY)
        elif classifier == 'logistic_l1':
            clf = sl.linear_model.LogisticRegression(penalty='l1')
            # Train with training data and crossvalidation
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=c_scale)},
                    nb_iter=nb_tr_iter,datname=kw)
            # Evaluate with test data
            print 'Report on Dev Data'
            print '-----------------------'                 
            # Evaluate with test data
            results = tp.classifier_eval(clf_trained,testX,testY)
        else:
            raise IOError('Classifier name not recognized')
        allresults[kw]=results

    # Print and store the average results
    avgresults = np.nanmean(allresults.values(),axis=0)
    avgresults_keys = ['avg_prec','avg_rec','avg_fscore','avg_acc','avg_AUC']
    allresults['avg_results'] = {akey:avgresults[i] for i,akey in\
        enumerate(avgresults_keys)}
    maxresults = np.nanmax(allresults.values(),axis=0)
    maxresults_keys = ['max_prec','max_rec','max_fscore',\
            'max_acc','max_AUC']
    allresults['max_results'] = {akey:maxresults[i] for i,akey in\
        enumerate(maxresults_keys)}
    print allresults['avg_results']
    print allresults['max_results']
    print 'Computation Time:',time.time()-old_time

    # Store all the important information
    allresults['best_classifier']=clf_trained
    allresults['classifier_type']=classifier
    allresults['c_scale']=c_scale
    allresults['data_normalizer']=normalizer
    allresults['scale_rating']=scale_rating
    allresults['modalities_used']=modality
    allresults['lowerthresh_Y']=lowerthresh_Y
    allresults['upperthresh_Y']=upperthresh_Y

    # Put a suitable filename and store allresults
    resultfile = 'results_{0}_{1}_{2}_{3}_{4}_{5}.pkl'.format(classifier,\
        c_scale,scale_rating,''.join([m[0] for m in modality]),\
        lowerthresh_Y,upperthresh_Y)
    resultfile = os.path.join(ted_data_path,'TED_stats/'+resultfile)
    cp.dump(allresults,open(resultfile,'wb'))

def put_in_bluehive():
    '''
    Unimportant code to submit job in Bluehive
    '''
    # Run 1
    # params=[{'classifier':'logistic_l1','c_scale':10.,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':2.,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':1.,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':10.,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':2.,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':1.,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.5,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.1,'nb_tr_iter':100},
    # {'classifier':'logistic_regression','c_scale':10.,'nb_tr_iter':100},
    # {'classifier':'logistic_regression','c_scale':2.,'nb_tr_iter':100},
    # {'classifier':'logistic_regression','c_scale':1.,'nb_tr_iter':100},
    # {'classifier':'logistic_regression','c_scale':0.5,'nb_tr_iter':100},
    # {'classifier':'logistic_regression','c_scale':0.1,'nb_tr_iter':100}]

    # Run 2
    # params=[
    # {'classifier':'logistic_l1','c_scale':1.75,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':1.35,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':1.15,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.85,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.65,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.45,'nb_tr_iter':100},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':1.75,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':1.35,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':1.15,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.85,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.75,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.65,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.45,'nb_tr_iter':100},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100}]

    # Run 3
    # params=[
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.25,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']}]
    
    # Run 4
    # params=[
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']}]

    # Run 5
    # params=[
    # {'classifier':'logistic_l1','c_scale':0.0001,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0010,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0050,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0100,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0500,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.1000,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.2500,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0001,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0005,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0010,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0050,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0100,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0500,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.1000,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.2500,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.0001,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.0010,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.0050,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.0100,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.0500,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.1000,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.2500,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'LinearSVM','c_scale':0.0001,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.0010,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.0050,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.0100,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.0500,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.1000,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'LinearSVM','c_scale':0.2500,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.}]    

    # Run 6
    # params=[
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.005,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.001,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'LinearSVM','c_scale':0.0005,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']}]

    # Run 7
    # params=[
    # {'classifier':'logistic_l1','c_scale':2.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':2.2500,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':2.0000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.7500,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.0000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.1000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0100,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0010,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':0.0001,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':2.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':2.2500,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':2.0000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.7500,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.0000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.1000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0100,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0010,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.0001,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':2.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':2.2500,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':2.0000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':1.7500,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':1.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':1.0000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.5000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.1000,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.0100,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.0010,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.0001,'nb_tr_iter':100,'scale_rating':False,'lowerthresh_Y':50.,'upperthresh_Y':50.},
    # {'classifier':'logistic_l1','c_scale':0.5000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.0000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.5000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.7500,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':1.9000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':2.0000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':2.2500,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':2.5000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':3.0000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':10.,'upperthresh_Y':90.},
    # {'classifier':'logistic_l1','c_scale':0.5000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.0000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.5000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.7500,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':1.9000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':2.0000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':2.2500,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':2.5000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.},
    # {'classifier':'logistic_l1','c_scale':3.0000,'nb_tr_iter':100,'scale_rating':True,'lowerthresh_Y':30.,'upperthresh_Y':70.}]

    # Run 8
    # params = [
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':10.,'upperthresh_Y':90.,'modality':['pose','face','trajectory','audio']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['face','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['pose','trajectory','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['pose','face','audio','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['pose','face','trajectory','lexical']},
    # {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'lowerthresh_Y':30.,'upperthresh_Y':70.,'modality':['pose','face','trajectory','audio']}]

    # Run 9
    # params = [
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['lexical']},
    # {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['pose']},
    # {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['face']},
    # {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['trajectory']},
    # {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['audio']},
    # {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['lexical']}]

    # Run 10
    params = [
    {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.01,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.05,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.05,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.05,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.05,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.05,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.1,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.15,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.15,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.15,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.15,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.15,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.25,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.5,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.0,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.25,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.25,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.25,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.25,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.25,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':1.5,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':0.75,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['pose'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['face'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['trajectory'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['audio'],'lowerthresh_Y':10.,'upperthresh_Y':90.},
    {'classifier':'logistic_l1','c_scale':2.0,'nb_tr_iter':100,'modality':['lexical'],'lowerthresh_Y':10.,'upperthresh_Y':90.}]
    
    if not 'SLURM_ARRAY_TASK_ID' in os.environ:
        print 'Must run as job array in Bluehive'
        return
    print 'Job Index:',os.environ['SLURM_ARRAY_TASK_ID']
    print 'Parameter_Length:',len(params)
    classify_multimodal(**params[int(os.environ['SLURM_ARRAY_TASK_ID'])])


if __name__=='__main__':
    put_in_bluehive()

####### Methods below this line are not ready for new code structure #############
def regress_ratings(scores,Y,regressor='SVR',cv_score=sl.metrics.r2_score):
    '''
    Try to predict the ratings using regression methods. Besides training
    the regressors, it also evaluates them.

    Use the following command to get the initial arguments:
    scores,Y,_,_ = __loaddata__()
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
            print 'Report on Dev Data:'
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
            print 'Report on Dev Data:'
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
            print 'Report on Dev Data:'
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
            print 'Report on Dev Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)
    

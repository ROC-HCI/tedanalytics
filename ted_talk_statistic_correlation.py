import cPickle as cp
import os
import nltk
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from list_of_talks import all_valid_talks
from TED_data_location import ted_data_path

# Returns indices with outliers removed
def remove_outlier(alist):
    nplist = np.array(alist)
    mean = np.mean(nplist)
    std = np.std(nplist)
    idx = np.where((nplist<mean+3.*std) & (nplist>mean-3.*std))
    print 'Outlier Removal'
    print 'input list length:',len(alist)
    print 'output list length:',len(idx[0])
    print 'outliers:',len(alist)-len(idx[0])
    print
    return idx[0]

# abs_ratcnt = Absolute Rating Count
def plot_correlation(abs_ratcnt,infolder,
        outfolder,show_scatter=False):
    alltalks = [str(afile)+'.pkl' for afile in all_valid_talks]
    tottalks = len(alltalks)
    totlen,totut,tottok,totsent = 0,0,0,0
    lenlst,viewlst,ratinglst,topratings,timealive,kwlst=[],[],{},{},[],[]
    agelist=[]
    titles=[]
    allratings={}
    allrating_names=['beautiful','confusing','courageous','fascinating','funny',\
            'informative','ingenious','inspiring','jaw-dropping','longwinded',\
            'obnoxious','ok','persuasive','total_count','unconvincing']
    # Reading all the pickle files and enlisting required info
    for afile in alltalks:
        print afile
        atalk=cp.load(open(infolder+afile,'rb'))
        # View count
        viewlst.append(atalk['talk_meta']['totalviews'])
        # Age of talk
        assert type(atalk['talk_meta']['datecrawled'])==datetime.datetime and \
            type(atalk['talk_meta']['datepublished'])==datetime.datetime,\
            'Talk age calulation: datetime type mismatch'
        age_days = (atalk['talk_meta']['datecrawled']-atalk['talk_meta']['datepublished']).days
        agelist.append(age_days)
        # Update total ratings and list the highest rating of each talk
        for akey in allrating_names:
            if akey=='total_count':
                continue
            if not allratings.get(akey):
                if not abs_ratcnt:
                    allratings[akey]=[atalk['talk_meta']['ratings'].get(akey,0)/
                    float(atalk['talk_meta']['ratings']['total_count'])]
                else:
                    allratings[akey]=[atalk['talk_meta']['ratings'].get(akey,0)]
            else:
                if not abs_ratcnt:
                    allratings[akey].append(float(atalk['talk_meta']['ratings'].get(akey,0))/\
                        float(atalk['talk_meta']['ratings']['total_count']))
                else:
                    allratings[akey].append(float(atalk['talk_meta']['ratings'].get(akey,0)))
    # Drawing the scatter plots
    allcorr=[]
    for ind,akey in enumerate(allratings):
        corr_ages  = np.corrcoef(agelist,allratings[akey])[0,1]
        print 'Correlation coefficient for rating',akey,'and Age of Video:',corr_ages
    print
    for ind,akey in enumerate(allratings):
        # remove the outliers because some ratings are so high that it skews the plot
        #idx = remove_outlier(allratings[akey])
        #x = [viewlst[i] for i in idx]
        #y = [allratings[akey][i] for i in idx]
        # Calculate Correlation Coefficient
        corr_views = np.corrcoef(viewlst,allratings[akey])[0,1]
        allcorr.append(corr_views)
        print 'Correlation coefficient for rating',akey,'and view count:',corr_views
        if show_scatter:
            plt.figure(ind)
            plt.scatter(viewlst,allratings[akey],alpha=0.33)
            # plot the x axis in log scale
            plt.xscale('log',nonposy='clip')
            plt.xlabel('Total Viewcount (log scale)')
            plt.ylabel('Percent of ratings')
            plt.title('rating-view scatter plot '+akey+\
                'corr: {:0.2f}'.format(z))
            plt.tight_layout()
            if abs_ratcnt:
                plt.savefig(outfolder+'scatter_'+akey+'absolute'+'.eps')
            else:
                plt.savefig(outfolder+'scatter_'+akey+'.eps')

    plt.figure()
    allcorr = np.array(allcorr)
    idx = np.argsort(allcorr)
    pos = list(range(len(allcorr)))
    plt.barh(pos,allcorr[idx])
    plt.yticks(pos,[allratings.keys()[i] for i in idx])
    plt.xlabel('Correlation Coefficient')
    plt.title('CorrCoef of Ratings vs. Total View')
    plt.tight_layout()
    if abs_ratcnt:
        plt.savefig(outfolder+'CorrCoef of Ratings vs. Total View (absolute)'+'.eps')
    else:
        plt.savefig(outfolder+'CorrCoef of Ratings vs. Total View'+'.eps')

if __name__=='__main__':
    plot_correlation(False,
            infolder=os.path.join(ted_data_path,'TED_meta/'),
            outfolder=os.path.join(ted_data_path,'TED_stats/'))

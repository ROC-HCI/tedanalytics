import os
import re
import csv
import urllib2
import sys
import json
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from time import sleep
import cPickle as cp
from subprocess import call

"""
The older crawler is not usable because the TED talk website is
changed recently (As seen in Oct 20th, 2017). Therefore, we need a 
new crawler. In the new system, the transcripts are timestamped
per paragraph, not per utterance. Also, the meta data contains
an additional JSON containing the complete meta data. Other
aspects are tried to keep backward compatible.

The crawler automatically downloads the videos unlike the previous crawler
"""

def get_trans_new(src_url):
    '''
    Get the transcripts from the new format (as of Aug 16, 2017) of the
    TED talk web pages.
    '''
    talk_text = ''
    resp = urllib2.urlopen(src_url+'/transcript?language=en')
    web_src = resp.read().decode('utf8','ignore').replace('\r',' ').replace('\n', ' ')
    text_seg = BeautifulSoup(web_src, 'lxml')
    time_divs = text_seg.find_all('div',
            {'class':' Grid__cell w:1of6 w:1of8@xs w:1of10@sm w:1of12@md '})
    text_divs = text_seg.find_all('div',
     {'class':' Grid__cell w:5of6 w:7of8@xs w:9of10@sm w:11of12@md p-r:4 '})
    # Store the time
    trns_micsec = []
    for atime in time_divs:
        mins,secs = atime.contents[1].contents[0].strip().split(':')
        trns_micsec.append((int(mins)*60+int(secs))*1000)
    # Store the text
    trns_text=[]
    for atext in text_divs:
        trns_text.append([aspan.strip() for aspan in re.split('\t*',
            atext.contents[1].contents[0])
            if aspan.strip()])
    return trns_text,trns_micsec

def get_meta_new(url_link):
    '''
    This is the function to extract the meta information from
    the new format (as of Aug 16, 2017) of TED talk web pages.
    '''
    # Retrieve and parse html
    resp = urllib2.urlopen(url_link)
    web_src = resp.read().decode('utf8','ignore').replace('\r',' ').replace('\n', ' ')
    text_seg = BeautifulSoup(web_src, 'lxml')

    # Identify correct block for next piece of information
    scripts = text_seg.find_all('script')
    for ascript in scripts:
        if not ascript.getText().startswith('q("talkPage.init"'):
            continue
        # Get the JSON containing information about the talk
        fullJSON =json.loads(re.search('(?<=q\(\"talkPage\.init\"\,\s)(\{.*\})',
            ascript.contents[0]).group(0))['__INITIAL_DATA__']
        # ID of the current talk
        talk_id = fullJSON['current_talk']
        currenttalk_JSON=None
        # Identify the JSON part for the current talk
        for atalk in fullJSON['talks']:
            if not atalk['id'] == talk_id:
                continue
            else:
                currenttalk_JSON = atalk
                break
        # Make sure that currenttalk_JSON is not none
        assert currenttalk_JSON, IOError('JSON detail of the talk is not found')

        ################## Extract all the necessary components ################
        # Get title
        title = currenttalk_JSON['title']
        # Get Author
        author=''
        for a_speaker in currenttalk_JSON['speakers']:
            author = author+a_speaker['firstname']+'_'+a_speaker['lastname']+';'
        # Get Keyword
        keywrds = currenttalk_JSON['tags']
        # Duration
        vidlen = currenttalk_JSON['duration']
        # Get the ratings as JSON string
        ratingJSON = currenttalk_JSON['ratings']
        ratings={}
        totcount=0
        for item in ratingJSON:
            ratings[str(item['name']).lower()]=item['count']
            totcount+=item['count']
        ratings['total_count']=totcount
        # Date Crawled
        datecrawl = datetime.now()
        # Download link
        if 'media' in fullJSON and 'internal' in fullJSON['media'] and \
                'podcast-regular' in fullJSON['media']['internal']:
            downlink = fullJSON['media']['internal']['podcast-regular']['uri']
        elif 'media' in fullJSON and 'internal' in fullJSON['media'] and \
            len(fullJSON['media']['internal'].keys()) > 0:
            # If the regular podcast link is not available 
            # save whatever is available
            linktype = fullJSON['media']['internal'].keys()[0]
            downlink = fullJSON['media']['internal'][linktype]['uri']
        else:
            downlink=''
        # Date published and Date Filmed
        for player_talk in currenttalk_JSON['player_talks']:
            datepub=-1
            datefilm=-1
            if player_talk['id']==talk_id:
                datepub = player_talk['published']
                datefilm = player_talk['published']
                break
        assert datepub is not -1 and datefilm is not -1,'Could not extract datepub or datefilm'
        # datepub = np.datetime64(
        #         currenttalk_JSON['speakers'][0]['published_at']).astype('O')
        # datefilm = np.datetime64(currenttalk_JSON['recorded_at']).astype('O')
        datepub = datetime.fromtimestamp(datepub)
        datefilm = datetime.fromtimestamp(datefilm)
         # Total views
        totviews = fullJSON['viewed_count']
        #########################################################################
        break
    return {'ratings':ratings,'title':title,'author':author,'keywords':keywrds,
    'totalviews':totviews,'downloadlink':downlink,'datepublished':datepub,
    'datefilmed':datefilm,'datecrawled':datecrawl,'vidlen':vidlen,'id':int(talk_id),
    'alldata_JSON':json.dumps(fullJSON)}

def crawl_and_update(csvfilename,videofolder,outfolder='./talks',runforrow=-1):
    '''
    Crawls the TED talks and extracts the relevant information.
    '''
    # Talk ID's to skip
    if os.path.isfile('to_skip.txt'):
        with open('to_skip.txt','rb') as f:
            toskip=[int(id) for id in f]
    # Build a list of urls to skip: all successes and failures
    # This is to skip a talk without actually visiting them
    toskip_url=[]
    if os.path.isfile('./success.txt'):
        with open('./success.txt') as f:
            toskip_url.extend([aurl.strip() for aurl in f])
    if os.path.isfile('./failed.txt'):
        with open('./failed.txt') as f:
            toskip_url.extend([aurl.strip() for aurl in f])
    toskip=set(toskip)
    toskip_url=set(toskip_url)

    # New style csv file
    with open(csvfilename,'rU') as f:
        csvfile = csv.DictReader(f)
        for rownum,arow in enumerate(csvfile):
            if runforrow is not -1 and rownum >= runforrow*100 \
                and rownum < (runforrow+1)*100:
                continue
            # Random waiting up to 10 sec
            sleep(int(np.random.rand(1)[0]*10.))
            url = arow['public_url']
            # Skip if already tried (succeded or failed)
            if url.strip() in toskip_url:
                continue
            # Try to download if not tried before
            # try:
            meta = get_meta_new(url)
            id_ = meta['id']
            print 'examining ...',id_,url
            # Skip if it is supposed to skip
            if id_ in toskip:
                print '... skipping'
                continue
            target_filename = os.path.join(outfolder,str(id_)+'.pkl')
            if os.path.isfile(target_filename):
                # Update the metadata if the talk already exists
                alldata = cp.load(open(target_filename))
                alldata['talk_meta']=meta
                cp.dump(alldata,open(target_filename,'wb'))
            else:
                # otherwise, try to save everything
                try:
                    txt,micstime = get_trans_new(url)
                    cp.dump({'talk_transcript':txt,'transcript_micsec':micstime,
                        'talk_meta':meta},open(target_filename,'wb'))
                except:
                    print 'Transcript not found for,',id_
                    # Not being able to find transcript means a failure
                    with open('./failed.txt','a') as ferr:
                        ferr.write(url+'\n')
                    continue
            # Now save the video
            target_videofile = os.path.join(videofolder,str(id_)+'.mp4')
            if os.path.exists(target_videofile):
                print 'Video exists. skipping ...'
                # Record Successes
                with open('./success.txt','a') as fsucc:
                    fsucc.write(url+'\n')
                continue
            print 'Video downloader started'
            if meta['downloadlink']:
                call(['wget','-O',target_videofile,meta['downloadlink']])
            else:
                print 'Video could not save. No link found',id_
            # Record Successes
            with open('./success.txt','a') as fsucc:
                fsucc.write(url+'\n')
            sys.stdout.flush()
            # except:
            #     # Record Failures
            #     with open('./failed.txt','a') as ferr:
            #         ferr.write(url+'\n')

if __name__=='__main__':
    if 'SLURM_ARRAY_TASK_ID' in os.environ: 
        crawl_and_update(
            './TED Talks as of 08.04.2017.csv',
            '/scratch/mtanveer/TED_video',
            os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        crawl_and_update(
            './TED Talks as of 08.04.2017.csv',
            '/scratch/mtanveer/TED_video')



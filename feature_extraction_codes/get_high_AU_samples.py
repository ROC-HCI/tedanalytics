import csv
import glob
import heapq

sample_list = {}
frame_number = {}
keylist=[' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c',
     ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c',
     ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU45_c']

for afile in glob.glob('./*openface.csv'):
    with open(afile) as f:
        reader = csv.DictReader(f)
        for arow in reader:
            for akey in keylist:
                if float(arow[akey])==1. and 2 < float(arow[akey.replace('c','r')]) > sample_list.setdefault(akey,0.):
                    sample_list[akey] = float(arow[akey.replace('c','r')])
                    frame_number[akey] = int(arow['frame'])
        if len(sample_list)==17:
            print afile
            print 'Intensity:',sample_list
            print 'frame_number:',frame_number
        else:
            sample_list = {}
            frame_number = {}
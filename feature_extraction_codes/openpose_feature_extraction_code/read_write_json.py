import json
import os
import glob
from pprint import pprint


#list_video = []
#videoListing = os.listdir("./TED_Cropped/")
#for item in videoListing:
#    if ".mp4" in item:
#        list_video.append(item[:-4])
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v")
args = parser.parse_args();

video = (args.v).replace("/media/nvidia/Seagate1/TED/TED_Cropped/","").replace(".mp4","")
# print(video)
i = 0
result = []
for f in sorted(glob.glob("./output/%s*.json"%video)):
    with open(f) as json_file:
        print(f)
        json_decoded = json.load(json_file)
        json_decoded['frame'] = i
        result.append(json_decoded)
    i = i+1    
data_file = './json_output/%s.json'%video
with open(data_file, 'w') as outfile:
    json.dump(result, outfile)

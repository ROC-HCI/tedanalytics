import cPickle as pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity 
import glob
import csv
import os
import numpy as np

threshold_val = 0.65
confidence = 0.8
openface_dir="/localdisk/TED_feature/openface_features_all/"
output_folder = "/p/google-glass/openface_sentencewise_features/"
meta_location="/p/behavioral/TED_dataset/TED_meta"
face_vec_dir="/p/behavioral/TED_dataset/TED_feature_openface/face_vector/"
sentence_boundary_dir="/p/behavioral/TED_dataset/TED_feature_sentence_boundary/"

def is_present(frame,vec,vec_list,vec_val,frame_list):
	if not vec_list:
		return False

	for i,v in enumerate(vec_list):
		cos_sim =  cosine_similarity([vec.tolist()],[v.tolist()])[0][0]
		if cos_sim > threshold_val:
			vec_val[i]=vec_val[i]+1
			frame_list[i].append(frame)
			return True

	return False


def get_presenter_frames(pkl_file):

	vec_list=[]
	vec_val=[]
	frame_list = []
	with open(pkl_file,"rb") as fp:
		features=pickle.load(fp)
		vectors=features[0]
		shape=features[1]
	for frame,vec in vectors.items():
		if vec is None:
			continue
		if not is_present(frame,vec,vec_list,vec_val,frame_list):
			vec_list.append(vec)
			vec_val.append(0)
			frame_list.append([frame])

	if not vec_val:
		return []

	i=vec_val.index(max(vec_val))
	return frame_list[i]

def get_sen_frame_num(data,interval):
	try:
		data=data[data[:,1]>=interval[0]]
		data=data[data[:,1]<=interval[1]]
		turn_frame=np.take(data,0,axis=1)
		turn_frame=turn_frame.astype(int)
		return turn_frame.tolist() 
	except:
		pass

	return list()



def get_openface_frames(video_num,frame_list,sen_time_intervals):
	sentence_frame_num_list=[]
	try:
		frame_list=np.array(frame_list)
		frame_list=frame_list-1
		f_name=openface_dir+str(video_num)+"_openface.csv"
		data=np.loadtxt(f_name,delimiter=',',skiprows=1)
		data=np.take(data,frame_list,axis=0)
		data=data[data[:,3]==1]
		data=data[data[:,2]>=confidence]
		data=np.take(data,[0,1],axis=1)
		for interval in sen_time_intervals:
			sen_frames=get_sen_frame_num(data,interval)
			sentence_frame_num_list.append(sen_frames)
	except:
		pass

	return sentence_frame_num_list


def get_openface_frames_num_per_sentence(video_num):
	sentence_frame_num_list=[]
	try:
		cmu_openface_file=face_vec_dir+str(video_num)+".pkl"
		tadas_openface_file=openface_dir+str(video_num)+"_openface.csv"
		sentence_boundary_file=sentence_boundary_dir+str(video_num)+'.pkl'
		
		frame_list=get_presenter_frames(cmu_openface_file)

		if not frame_list:
			return list()

		sen_time_intervals=[]

		with open(sentence_boundary_file,"rb") as fp:
			sen_boundaries=pickle.load(fp)
			sen_boundaries=sen_boundaries['sentences']

			for bound in sen_boundaries:
				s=bound['beg_time']
				e=bound['end_time']
				if s and e:
					sen_time_intervals.append([float(s),float(e)])

		if not sen_time_intervals:
			return list()
		sentence_frame_num_list=get_openface_frames(video_num,frame_list,sen_time_intervals)

	except:
		pass
	return sentence_frame_num_list		
	
for afile in glob.glob(os.path.join(meta_location,'*.pkl')):
        
	path,filename = os.path.split(afile)
	print 'Processing:',filename
	sentence_frame_num_list=get_openface_frames_num_per_sentence(int(filename[:-4]))
	outfile = os.path.join(output_folder,filename[:-4]+'.pkl')
	print 'output to:',outfile
	pickle.dump(sentence_frame_num_list,open(outfile,'wb'))


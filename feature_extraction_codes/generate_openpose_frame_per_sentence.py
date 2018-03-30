import cPickle as pickle
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import glob
import csv
import numpy as np
import json
import matplotlib.pyplot as plt

threshold_val = 0.65
# batch=int(sys.argv[2])
metafolder='/p/behavioral/TED_dataset/TED_meta/*.pkl'
outfolder='/p/google-glass/openpose_sentencewise_features'
openface_dir="/localdisk/TED_feature/openface_features_all/"
face_vec_dir="/p/behavioral/TED_dataset/TED_feature_openface/face_vector/"
openpose_dir="/p/behavioral/TED_dataset/TED_feature_openpose/All_JSON/"
sentence_boundary_dir="/p/behavioral/TED_dataset/TED_feature_sentence_boundary/"
single_person_talks=[]


f = open('/p/behavioral/TED_dataset/misc/single_person_talks.txt', 'r')
data = f.readlines()
for row in data:
	row=row.split(',')
	row=row[2].split('.')
	single_person_talks.append(row[0].strip())

fps_map={}

f = open('/p/behavioral/TED_dataset/misc/fps.txt', 'r')
data = f.readlines()
for row in data:
	row=row.split(' ')
	v_key=row[0].split('.')
	v_key=v_key[0].strip(' \t\n\r')
	fps_map[v_key]=int(row[1].strip(' \t\n\r'))



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
	return frame_list[i],shape

def get_open_face_face_cord(root,frame_num):
	f_name=openface_dir+str(root)+"_openface.csv"
	data=np.loadtxt(f_name,delimiter=',',skiprows=1)
	r=data[frame_num]
	r=r.tolist()
	x_cor=r[298:366]
	y_cor=r[366:434]
	return x_cor,y_cor


def draw_openpose_features(root,frame):
	f_name=openpose_dir+str(root)+".json"
	with open(f_name) as json_data:
	    d = json.load(json_data)
	    print d[frame]['frame']
	    people=d[frame]['people']
	    for p in people:
	    	x=p['pose_keypoints']
	    	print x
	    	point_x = [item*1.25 for i,item in enumerate(x) if i%3==0]
	    	point_y = [item*1.25 for i,item in enumerate(x) if i%3==1]
	    	conf = [item for i,item in enumerate(x) if i%3==2]
	    	for i,j,k in zip(point_x,point_y,range(len(point_x))):
	    		plt.scatter(i,-j)
	    		plt.text(i,-j,str(k))


	    print "*******************"
	    x_cor,y_cor= get_open_face_face_cord(root,390)
	    print x_cor
	    print y_cor
	    for i,j,k in zip(x_cor,y_cor,range(len(x_cor))):
    		plt.scatter(i,-j)

    	#plt.axis('ij')
    	plt.show()

#avoid confidence < 0.2
def get_normalized_pose_point(pose_keypoints,shape):
	point_x = [item/float(shape[1]) for i,item in enumerate(pose_keypoints) if i%3==0]
	point_y = [item/float(shape[0]) for i,item in enumerate(pose_keypoints) if i%3==1]
	conf=[item for i,item in enumerate(pose_keypoints) if i%3==2]
	ref_x=point_x[1]
	ref_y=point_y[1]

	norm_x,norm_y=[],[]
	for i,p_x in enumerate(point_x):
		if conf[i]<0.2 or ( point_x ==0 and point_y ==0):
			norm_x.append(None)
			norm_y.append(None)
		else:
			norm_x.append(point_x[i]-ref_x)
			norm_y.append(point_y[i]-ref_y)

	return norm_x,norm_y


def get_openpose_features(root,frame_list,shape):
	try:
		f_name=openpose_dir+str(root)+".json"
		keypoints_x,keypoints_y=[],[]
		with open(f_name) as json_data:
		    data = json.load(json_data)
		    for d in data:
		    	frame=d['frame']
		    	people=d['people']
		    	if not people:
		    		continue
		    	if frame in frame_list or(frame>=frame_list[0] and frame<=frame_list[-1] and len(people)==1):
		    		p=people[0]
		    		pose_keypoints=p['pose_keypoints']
		    		norm_x,norm_y=get_normalized_pose_point(pose_keypoints,shape)
		    		keypoints_x.append(norm_x)
		    		keypoints_y.append(norm_y)
		    
		keypoints_x=np.array(keypoints_x,dtype=np.float)
		keypoints_y=np.array(keypoints_y,dtype=np.float)

		features=np.concatenate((keypoints_x,keypoints_y),axis=1)

		return features.tolist()
	except:
		pass

	return list()




def map_to_openpose_frame(root,frame_list):
	fps=fps_map[str(root)]
	frame_list=[((int(x/fps)-11) * 3)-1 for x in frame_list]
	frame_list=set(frame_list)
	return sorted(list(frame_list))


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
		data=np.take(data,[0,1],axis=1)
		for interval in sen_time_intervals:
			sen_frames=get_sen_frame_num(data,interval)
			sentence_frame_num_list.append(sen_frames)
	except:
		pass

	return sentence_frame_num_list




def get_openpose_frames_per_sentence(video_num):
	openpose_features_per_sentence=[]
	cmu_openface_file=face_vec_dir+str(video_num)+".pkl"
	tadas_openface_file=openface_dir+str(video_num)+"_openface.csv"
	sentence_boundary_file=sentence_boundary_dir+str(video_num)+'.pkl'
	
	frame_list,shape=get_presenter_frames(cmu_openface_file)

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
	sentence_frame_num_list=[map_to_openpose_frame(video_num,x) for x in sentence_frame_num_list]

	for sen_frames in sentence_frame_num_list:
		if sen_frames:
			features= get_openpose_features(video_num,sen_frames,shape)
			openpose_features_per_sentence.append(features)
		else:
			openpose_features_per_sentence.append([])

	return openpose_features_per_sentence

if __name__=='__main__':
	for afile in glob.glob(metafolder): 
	    path,filename = os.path.split(afile)
	    outfile = os.path.join(outfolder,filename[:-4]+'.pkl')
	    if os.path.exists(outfile):
	    	print 'skiping ...',filename
	    	continue
	    print 'Processing:',filename
	    try:
	    	openpose_features_per_sentence=get_openpose_frames_per_sentence(int(filename[:-4]))
	    except:
	    	continue
	    
	    print 'output to:',outfile
	    pickle.dump(openpose_features_per_sentence,open(outfile,'wb'))



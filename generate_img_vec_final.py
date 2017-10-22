import glob
import sys
import imageio
import openface
import numpy as np
import cPickle as pickle
from datetime import datetime
img_dim=96

def setup_openface_parameters():

        landmarks_path="openface/models/dlib/shape_predictor_68_face_landmarks.dat"
        netrwork_model_path="openface/models/openface/nn4.small2.v1.t7"
        align = openface.AlignDlib(landmarks_path)
        net = openface.TorchNeuralNet(netrwork_model_path, img_dim)
        return [align,net]



def get_vector(rgb_img,align,net):

        if rgb_img is None:
            raise Exception("Unable to load image:")

        # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        bb = align.getLargestFaceBoundingBox(rgb_img)
        if bb is None:
                raise Exception("Unable to find a face")

        aligned_face = align.align(img_dim,rgb_img, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align image")

        vec = net.forward(aligned_face)

        return vec



def get_openface_neural_net_features(video_file,align,net):
	reader=imageio.get_reader(video_file)
	vec_map={}
	shape=None
	is_read=True
	try:
        	for i,rgb_img in enumerate(reader):
			try:	
				if is_read and len(rgb_img)>0:
					shape=[len(rgb_img),len(rgb_img[0])]
					is_read=False
                		#print(i)
				vec=get_vector(rgb_img,align,net)
				vec_map[i]=vec
				#print(vec)
				
			except:
				#print("failed")
				vec_map[i]=None
			sys.stdout.flush()		
	except RuntimeError:
        	pass
	return [vec_map,shape]

	
def main():
	align,net=setup_openface_parameters()
	
	video_files_path="/scratch/mhasan8/TED_video/batch"+sys.argv[1]+"/*.mp4"
        log_file=open("/scratch/mhasan8/TED_features/openface/log/log"+sys.argv[1]+".txt","a")
        video_files=glob.glob(video_files_path)
	log_file.write(str(datetime.now()))
	for video_file in video_files:
		print(video_file)
		video_features=get_openface_neural_net_features(video_file,align,net)
		pkl_file="/scratch/mhasan8/TED_features/openface/pkl/"+video_file[video_file.rfind('/')+1:-4]+".pkl"
                with open(pkl_file,'wb') as f:
                        pickle.dump(video_features, f)

                log_file.write(str(datetime.now()))
                log_file.write(" video completed %s \n" % video_file[video_file.rfind('/')+1:])
	log_file.close()
main()

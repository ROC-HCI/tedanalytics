
# coding: utf-8

# In[137]:

import parselmouth
import numpy as np
import os
import cPickle as cp
import glob
import shlex
import subprocess
import scipy as sp
import scipy.stats as ss
import re


# In[11]:

data_path = "/scratch/echowdh2/TED_DATA/"


num_formant=5
num_thread = 30
num_file_per_thread=85


# In[139]:

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# In[5]:

def get_intensity_values_of_a_fav_row(beg_time,end_time,intensity):
    temp_array= []
    cur_time = beg_time
    while(cur_time < end_time):
        temp_array.append(intensity.get_value(cur_time))
        cur_time += intensity.get_time_step()
    return temp_array

    np_intensity = np.array(temp_array)

    return {'mean':np.nanmean(np_intensity),'std':np.nanstd(np_intensity)}
    


# In[6]:

def get_pitch_values_of_a_fav_row(beg_time,end_time,pitch):
    temp_array = []


    beg_frame_no = int(pitch.get_frame_number_from_time(beg_time))
    end_frame_no = int(pitch.get_frame_number_from_time(end_time))
    for i in range(beg_frame_no,end_frame_no+1):
        temp_array.append(pitch.get_value_in_frame(i))
    return temp_array    

    


# In[7]:

def get_formants_values_of_a_fav_row(beg_time,end_time,formant,formant_no=[1,2,3,4,5]):
    temp_mat= []
    for cur_formant_no in formant_no:
        cur_time = beg_time
        temp_array=[]
        while(cur_time < end_time):
            temp_array.append(formant.get_value_at_time(cur_formant_no,cur_time))
            cur_time += formant.get_time_step()
        temp_mat.append(temp_array) 
    return temp_mat


    
    


# In[104]:

def get_pitch_features(pitch,np_pitch):
    frac_voiced_frames = float(pitch.count_voiced_frames())/pitch.get_number_of_frames()
    pitch_dict =  {'mean':np.nanmean(np_pitch),'std':np.nanstd(np_pitch),'fraction_voiced':frac_voiced_frames,'min':np.nanmin(np_pitch),            'max':np.nanmax(np_pitch),'median':np.nanmedian(np_pitch),'25_percentile':np.nanpercentile(np_pitch,25),            '75_precentile':np.nanpercentile(np_pitch,75),'kurtosis':float(ss.kurtosis(np_pitch,nan_policy='omit')),             'skewness':float(ss.skew(np_pitch,nan_policy='omit'))}
    return pitch_dict  
    


# In[105]:

def get_intensity_features(intensity,np_intensity):
    
    intensity_dict =  {'mean':np.nanmean(np_intensity),'std':np.nanstd(np_intensity),'min':np.nanmin(np_intensity),'max':np.nanmax(np_intensity),'median':np.nanmedian(np_intensity),'25_percentile':np.nanpercentile(np_intensity,25),            '75_precentile':np.nanpercentile(np_intensity,75),'kurtosis':float(ss.kurtosis(np_intensity,nan_policy='omit')),             'skewness':float(ss.skew(np_intensity,nan_policy='omit'))}
    return intensity_dict 


# In[124]:

def get_formant_features(formant,np_formant):
    #print(temp_mat)
    mean = np.squeeze(np.asarray(np.nanmean(np_formant,axis=1)))
    std = np.squeeze(np.asarray(np.nanstd(np_formant,axis=1)))
    min_val = np.squeeze(np.asarray(np.nanmin(np_formant,axis=1)))
    max_val = np.squeeze(np.asarray(np.nanmax(np_formant,axis=1)))
    #median = np.squeeze(np.asarray(np.nanmedian(np_formant,axis=1)))
    #percent_25 = np.squeeze(np.asarray(np.nanpercentile(np_formant,25,axis=1)))
    #percent_75 = np.squeeze(np.asarray(np.nanpercentile(np_formant,75,axis=1)))
    kurtosis = np.squeeze(np.asarray((ss.kurtosis(np_formant,nan_policy='omit',axis=1))))
    skewness = np.squeeze(np.asarray((ss.skew(np_formant,nan_policy='omit',axis=1))))
    
    return {'mean':mean,'std':std,'min':min_val,'max':max_val,'kurtosis':kurtosis,'skewness':skewness}
    


# In[174]:

def audio_features_for_a_talk(talk_id,split_index):
    

    aac_file_path =  os.path.join(data_path,'TED_audio_aac/'+str(talk_id)+".aac")
    wav_file_path =  os.path.join(data_path,'My_code/tempWav_for_praat/'+str(split_index)+".wav")
    meta_file_path = os.path.join(data_path,'TED_meta/'+str(talk_id)+".pkl")

   

    rows = cp.load(open(meta_file_path))['fave_style_transcript']

    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)

    aac_to_wav_convert_command = "ffmpeg -i {0} {1}".format(aac_file_path,wav_file_path)
    args = shlex.split(aac_to_wav_convert_command)
    p_aac_to_wav = subprocess.Popen(args)
    p_aac_to_wav.wait()

    snd = parselmouth.Sound(wav_file_path)

    intensity = snd.to_intensity()
    pitch = snd.to_pitch_ac()
    formant = snd.to_formant_burg()
    

    all_pitch=[]
    all_intensity=[]
    all_formants=[]
    for i in range(num_formant):
        all_formants.append([])
    #print all_formants
    for a_row in rows:
        #print a_row
        beg_time = float(a_row['beg_time'])
        end_time = float(a_row['end_time'])
        #print beg_time,end_time
        all_pitch.extend(get_pitch_values_of_a_fav_row(beg_time,end_time,pitch))
        all_intensity.extend(get_intensity_values_of_a_fav_row(beg_time,end_time,intensity))
        new_formant = get_formants_values_of_a_fav_row(beg_time,end_time,formant)
        for i in range(num_formant):
            all_formants[i].extend(new_formant[i])
            
    np_pitch = np.array(all_pitch)
    pitch_feature_dict = get_pitch_features(pitch,np_pitch)
    #print pitch_feature_dict
    
    np_intensity = np.array(all_intensity)
    intensity_feature_dict = get_intensity_features(intensity,np_intensity)
    #print intensity_feature_dict
    
    np_formant = np.matrix(all_formants)
    formant_feature_dict= get_formant_features(formant,np_formant)
    #print formant_feature_dict
    
    entire_talk_dict = {'talk_id':talk_id,'pitch':pitch_feature_dict,'intensity':intensity_feature_dict,'formant':formant_feature_dict}
    return entire_talk_dict
  


def extract_audio_feature_for_a_split(split_index):
    #aac_file_name =  basic_file_name +".aac"
    #will need to change it in the bh python file
    aac_file_path =  os.path.join(data_path,'TED_audio_aac/')
    dict_of_all_talk_dict={}
    
    
    all_files = sorted(glob.glob(os.path.join(aac_file_path, '*.aac')),key = numericalSort)
    relevant_files = all_files[split_index*num_file_per_thread : min(len(all_files),(split_index+1)*num_file_per_thread)]
    for f in relevant_files:
        talk_id = int(f[f.rfind("/")+1:].replace(".aac",""))

        output_fp_path = os.path.join(data_path,'TED_audio_features/'+str(talk_id)+'.pkl')
        if os.path.exists(output_fp_path):
            continue
        meta_file_path = os.path.join(data_path,'TED_meta/'+str(talk_id)+".pkl")

        if os.path.exists(meta_file_path) == False:
            continue 

        output_fp = open(output_fp_path,'wb')
      

        talk_dict = audio_features_for_a_talk(talk_id,split_index)
        cp.dump(talk_dict,output_fp)
        output_fp.close()
        print "Completed: "+str(talk_id)
    
 

    
 
    
if __name__=='__main__':
   
    extract_audio_feature_for_a_split(split_index=int(os.environ['SLURM_ARRAY_TASK_ID']))
    #we are using 2 as a pilot case



import csv
import cPickle as cp
import os
import glob
import string
import re
import subprocess
import shlex
import nltk
from nltk.tokenize import sent_tokenize
from textgrid import *
data_path = "/scratch/echowdh2/TED_DATA/"
tot_meta_file = 2480
num_thread = 30
num_file_per_thread=2480


# In[ ]:




# In[28]:

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# In[29]:

def produce_time_boundary_for_all_sentence_in_all_files(split_index):
    'Given the name of the file name, produces a .pkl file with an approximate timestamp for each sentence.'
    #THe handler to the file we are outputting the data
    #pkl_output_path = os.path.join(data_path,'TED_fave_aligned/')
    pkl_input_path = os.path.join(data_path,'TED_meta/')
    removed_punctuations = "'\"#$%&()*+,/:;<=>@[\]^_`|~"
    
   
    all_files = sorted(glob.glob(os.path.join(pkl_input_path, '*.pkl')),key = numericalSort)
    relevant_files = all_files[split_index*num_file_per_thread : min(len(all_files),(split_index+1)*num_file_per_thread)]
    
    
    for input_file in relevant_files:
       
        #Get the fave style rows
        fave_styled_rows = cp.load(open(input_file))['fave_style_transcript']
        entire_text = ""

        #Get hold of the corresponding .WAV file path
        #First extract the base name of the file
        basic_file_name = input_file[input_file.rfind("/")+1:].replace(".pkl","")

        aac_file_name =  basic_file_name +".aac"
        #will need to change it in the bh python file
        aac_file_path =  os.path.join(data_path,'TED_audio_aac/'+aac_file_name)
        
        if(os.path.exists(aac_file_path) == False):
            continue
        
        #AT first, convert the aac file into a wav file and save it in tempWav folder
        wav_file_path =  os.path.join(data_path,'My_code/tempWav/'+str(split_index)+".wav")
        print wav_file_path
        #Now make the system call to convert into a wav file
        #HOwever, first we will need to remove the existing file
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)
        #NOw make a system call to make the conversion
        #sub_wav_linux_command = "ffmpeg -ss {0} -t {1} -i {2} -ar 11025 {3}".format(row_beg_time+1,row_end_time-row_beg_time,wav_file_path,"temp_sub_audio.wav")
        aac_to_wav_convert_command = "ffmpeg -i {0} {1}".format(aac_file_path,wav_file_path)
        args = shlex.split(aac_to_wav_convert_command)
        p_aac_to_wav = subprocess.Popen(args)
        p_aac_to_wav.wait()
    

        #wav_file_name =  input_file[input_file.rfind("/")+1:].replace(".pkl",".wav")
        #wav_file_path =  os.path.join(data_path,'TED_audio_wav/'+wav_file_name)
        
        #If the .wav file has not been loaded yet, just skip this iteration
        if(os.path.exists(wav_file_path) == False):
            continue

        #if we have already built the resulting .pkl file from sentence segmentation,skip this iteration
        cp_file_path = os.path.join(data_path,'TED_sentence_time_boundary/'+str(basic_file_name)+".pkl")
        if(os.path.exists(cp_file_path) == True):
            continue
        
    
        #THe beginning time of all segments
        all_row_beg_times = []
        #The corresponding faav id of all segments
        faav_id = []

        #First, we need to delete all the .TextGrid files in the appropriate folder
        text_grid_files = glob.glob(os.path.join(data_path,'My_code/tempAligned/'+str(split_index)+'*'))
        for f in text_grid_files:
            #base_name =  f[f.rfind("/")+1:]
            #if(int(base_name[0]) == split_index):
            os.remove(f)

        index = 0

        for i in range(len(fave_styled_rows)):
            row = fave_styled_rows[i]
            #print row.keys()
            #he sentences were in list, so we just made them into one string
            row['sentences'] = ' '.join(row['sentences'])
            entire_text += row['sentences']
            #print row['sentences']
            #This label is unncecessary
            del row['labels']
            #We are removing all the irrelevant punctuations
            table = string.maketrans(removed_punctuations," "*len(removed_punctuations))
            sentence_wo_punc = row['sentences'].translate(table)

            sentence_wo_punc = sentence_wo_punc.replace("."," FULL-STOP ")
            sentence_wo_punc = sentence_wo_punc.replace("!"," FULL-STOP ")
            sentence_wo_punc = sentence_wo_punc.replace("?"," FULL-STOP ")
            #To correctly pass through HTK, we will remove \t with whitspace
            sentence_wo_punc = sentence_wo_punc.replace("\t"," ")


            #Then splitting the sentence using space
            new_words = sentence_wo_punc.split()
            #We need to store it in a file "temporarily" for the future phases of the algo to process it
            final_words = ' '.join(new_words)
            #print "\n\n"
            #print final_words
            #print "\n\n"

            #Save the words in a temprarily file


            transcript_file = open("temp_transcript"+str(split_index)+".txt",'w')
            transcript_file.write(final_words+"\n\n")
            transcript_file.close() 

            #Now extract the row beginning and end time
            row_beg_time = row['beg_time']
            row_end_time = row['end_time']
            #We are storing all the row beginning times for future pre-processing. 1 is added for offset correction.
            all_row_beg_times.append(row_beg_time+1)
            faav_id.append(i)
            #/media/wasifur/LinuxBhai/TED_Data/Necessary_scripts/p2fa/p2fa
            #Now we will call the align function from the p2fa module it .
            #Now we will produce a sub-section of the corresponding .wav file 

            #Some observations: THe beg_time and end_time collide with each other. In my observation, the beginning time 
            #starting from the second row should be added to 1 for maximum correctness. The speaker starts speaking from one
            #second after the mentioned time and goes all the way to the end of the time

            #AT first set the appropriate file paths
            transcript_file_path = os.path.join(data_path,'My_code/temp_transcript'+str(split_index)+'.txt')
            sub_wav_file_path = os.path.join(data_path,'My_code/temp_sub_audio'+str(split_index)+'.wav')
            output_file_path = os.path.join(data_path,"My_code/tempAligned/"+str(split_index)+"fave_aligned_transcript"+str(index)+".TextGrid")
            align_code_file_path = os.path.join(data_path,'Necessary_scripts/p2fa/p2fa/align.py')

            index +=1
            #Before executing the commnads, we need to remove the previously stored files

            if os.path.exists(sub_wav_file_path):
                #print str(os.path.exists(sub_wav_file_path)) + " exists"
                os.remove(sub_wav_file_path)
                #print str(os.path.exists(sub_wav_file_path)) + " does not exist"
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
        



            sub_wav_linux_command = "ffmpeg -ss {0} -t {1} -i {2} -ar 11025 {3}".format(row_beg_time+1,row_end_time-row_beg_time,wav_file_path,"temp_sub_audio"+str(split_index)+".wav")
            print sub_wav_linux_command
            #print row_beg_time//60,row_beg_time%60
            #print row_end_time//60,row_end_time%60
            #print final_words
            args = shlex.split(sub_wav_linux_command)
            p = subprocess.Popen(args)

            #Now, we need to call the align.py file in the appropriate folder.
            p.wait()

            align_linux_command = "python {0} {1} {2} {3} {4}".format(align_code_file_path,sub_wav_file_path,transcript_file_path,output_file_path,split_index)
            print align_linux_command

            args1 = shlex.split(align_linux_command)
            p1 = subprocess.Popen(args1)
            p1.wait()
        prepare_time_segment_for_each_sentence(all_row_beg_times,entire_text,basic_file_name,faav_id,split_index)
        
        
        
        
    #pout = open(pkl_output_path, 'wb') 


# In[ ]:





def merge_all_intervals(beginning_times,split_index):
    text_grid_files = sorted(glob.glob(os.path.join(data_path,'My_code/tempAligned/'+str(split_index)+'*')),key = numericalSort)
    faav_id_of_each_interval=[]
    final_tg = None
    for i in range(len(text_grid_files)):
        f = text_grid_files[i]
        row_beginning_time = beginning_times[i]
        if(final_tg == None):
            final_tg = TextGrid()
            final_tg.read(f)
            final_tg_word_intervals = final_tg.getList("word")[0]
            new_interval_list = []
            #print "Before:"
            for interval in final_tg_word_intervals:
                #print interval
                interval = Interval(interval.minTime+row_beginning_time,interval.maxTime+row_beginning_time,interval.mark)
                new_interval_list.append(interval)
                faav_id_of_each_interval.append(i)
            #print "After:"
            final_tg_word_intervals.remove_all_intervals()
            #print final_tg_word_intervals
          
            
            for new_i in new_interval_list:
                final_tg_word_intervals.addIntervalSeq(new_i)
            
            
            
        else:
            temp_tg = TextGrid()
            temp_tg.read(f)
            temp_tg_word_intervals = temp_tg.getList("word")[0]
            for interval in temp_tg_word_intervals:
                nInterval = Interval(interval.minTime+row_beginning_time,interval.maxTime+row_beginning_time,interval.mark)
                final_tg_word_intervals.addIntervalSeq(nInterval)
                faav_id_of_each_interval.append(i)

        
    return final_tg_word_intervals,faav_id_of_each_interval
            


# In[36]:

#returns the index of all "FULL" and "STOP" as pair
def find_all_full_stops_in_interval(merged_interval_tiers,faav_id_of_each_interval):
    full_stops = []
    faav_id_of_each_sentence = []
    for i in range(len(merged_interval_tiers)):
        cur_int = merged_interval_tiers[i]
        
        if(cur_int.mark=="STOP"):
            if(i > 0 and merged_interval_tiers[i-1].mark == "FULL"):
                full_stops.append((i-1,i))
                faav_id_of_each_sentence.append(faav_id_of_each_interval[i])
            elif (i>1 and  merged_interval_tiers[i-1].mark == "sp" and  merged_interval_tiers[i-2].mark == "FULL"):
                full_stops.append((i-2,i))
                faav_id_of_each_sentence.append(faav_id_of_each_interval[i])

    return full_stops,faav_id_of_each_sentence
                
        


# In[37]:

def find_st_index_of_first_sentence(intervals):
    for i in range(len(intervals)):
        if(intervals[i].mark=="sp"):
            return i
        
        


# In[40]:

#We will merge all the textgrid files together. Now,all textgrid files start from time 0. So, we will need to 

def prepare_time_segment_for_each_sentence(beginning_times,all_text,wav_file_name,faav_ids,split_index):
    #print "starting to prepare time segment"
    all_sentences =  re.split('[.?!]',all_text)

    cp_file_name = os.path.join(data_path,'TED_sentence_time_boundary/'+str(wav_file_name)+".pkl")
    fp = open(cp_file_name,'wb')
    
    dict_of_all_sentences_list=[]
  
    
    #out_csv = open(os.path.join(data_path,'TED_sentence_time_boundary/'+str(wav_file_name)+".csv"), 'w')
    #writer = csv.DictWriter(out_csv, fieldnames = ["Start", "End", "Sentence"])
    #writer.writeheader()
    #writer.writerows([{'Start': 5, 'End': 6, 'Sentence': "good"}])
   
    merged_interval_tiers,faav_id_of_each_interval = merge_all_intervals(beginning_times,split_index)
    all_full_stop_indices,faav_id_of_each_sentence = find_all_full_stops_in_interval(merged_interval_tiers,faav_id_of_each_interval)
    st_index_of_first_sentence = find_st_index_of_first_sentence(merged_interval_tiers)
     
   
    cur_start = merged_interval_tiers[st_index_of_first_sentence].minTime
    cur_stop=-1
    index = 0
    for (full_index,stop_index) in all_full_stop_indices:
        cur_stop = merged_interval_tiers[full_index].minTime 
        #writer.writerows([{'Start': cur_start, 'End': cur_stop, 'Sentence': str(all_sentences[index])}])
        t_dict = {}
        t_dict['beg_time']=cur_start
        t_dict['end_time']=cur_stop
        t_dict['sentence']=all_sentences[index]
        t_dict['fav_id'] = faav_id_of_each_sentence[index]
        dict_of_all_sentences_list.append(t_dict)
        
        #print str(all_sentences[index])
        cur_start = merged_interval_tiers[stop_index-1].maxTime
        index +=1
    
    final_dict = {"sentences":dict_of_all_sentences_list}
    cp.dump(final_dict,fp)
    fp.close()
 
        
  


# In[ ]:






if __name__=='__main__':
    '''
    Given the path to TED_meta folder, generates all the
    transcripts and the corresponding talk id.
    '''
    #if you want time boundary for a particular file, then write the files name in the format like '/2.pkl', otherwise
    # make file_name = '/' if you want the output from all the files in the directory.
    produce_time_boundary_for_all_sentence_in_all_files(split_index=int(os.environ['SLURM_ARRAY_TASK_ID']))
    #we are using 2 as a pilot case





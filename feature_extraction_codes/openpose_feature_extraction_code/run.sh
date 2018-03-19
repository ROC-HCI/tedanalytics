 #Trimming video
VIDEOFILE='./video/*.mp4'
for f in $VIDEOFILE
do 
 videoname=$(basename $f)
 ffmpeg -y -i $f -c copy -ss 00:00:11 ./data/trimmed_video/$videoname
done

#Reduce Frame Rate
TRIMMEDFILE='./data/trimmed_video/*.mp4'
for f in $TRIMMEDFILE
do 
 videoname=$(basename $f)
 ffmpeg -y -i $f -vf "setpts=1.25*PTS" -strict -2 -r 3 ./TED_Cropped/$videoname 
done

FILENAME='/media/nvidia/Seagate1/TED/TED_Cropped/*.mp4'

#Print out Video Frame Size  
./video_resolution.sh 2>&1 | tee list_resolution.txt

#Print out all json file of openpose

for f in $FILENAME
do
 cd /home/nvidia/openpose
 ./build/examples/openpose/openpose.bin --video $f --write_keypoint_json /media/nvidia/Seagate1/TED/output -net_resolution 256x192
 
 cd /media/nvidia/Seagate1/TED
 python read_write_json.py -v $f
 rm ./output/*.json
done

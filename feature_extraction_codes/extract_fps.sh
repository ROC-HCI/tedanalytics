cd /p/behavioral/TED_dataset/TED_video
for i in *.mp4; do echo $i `ffmpeg -i $i 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"`; done >> fps.txt


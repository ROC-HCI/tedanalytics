FILENAME=./TED_Cropped/*.mp4
for f in $FILENAME
do
 eval $(ffprobe -v error -of flat=s=_ -select_streams v:0 -show_entries stream=height,width $f)
 size=${streams_stream_0_width}x${streams_stream_0_height}
 echo $(basename $f) $size
done

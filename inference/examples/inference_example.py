from inference import *


video_path = "/path/video.mp4"
video_path = "/home/celine/Videos/vlc-record-2024-03-21-11h33m32s-video81.mp4-.mp4"
#video_path = 0
output_json_path = '/path/result.json'
output_json_path = "/home/celine/testest44.json"

output_video_path = '/path/video_result3.mp4'
output_video_path = "/home/celine/Videos/vlc-record-result.mp4" 

if __name__ == "__main__":

    test = VideoInference(video_path=video_path)
    test.add_model(['yolov5', 'oc_sort', 'af_link'])

    #print(test.pipeline)

    #output_json can be None
    test.process_video()#_online()#output_json=output_json_path)

    test.visualization(video_path=output_video_path)


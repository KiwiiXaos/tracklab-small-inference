from inference import *


video_path = "/home/celine/Videos/vlc-record-2024-03-21-11h33m32s-video81.mp4-.mp4"
video_path = 0
#output_json_path = '/path/result.json'
output_json_path = "/home/celine/test1.json"

#output_video_path = '/path/video_result.mp4'
output_video_path = "/home/celine/Videos/vlc-record-result2.mp4" 

if __name__ == "__main__":

    test = VideoInference(video_path=video_path)
    test.add_model(['openpifpaf', 'oc_sort', 'af_link'])

    #print(test.pipeline)

    test.process_video_online()



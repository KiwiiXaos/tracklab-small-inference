from inference import *

folder_path = "/path/video.mp4"

if __name__ == "__main__":
    
    test = VideoInference()
    test.add_model(['yolov5', 'bytetrack_sort'])
    test.process_folder(directory_path= folder_path, visualization = False)


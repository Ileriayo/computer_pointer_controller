'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import os
import cv2
from numpy import ndarray
import logging

log = logging.getLogger(__name__)

class InputFeeder:
    def __init__(self, input_file, input_type=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_file = input_file
        self.input_type = input_type
        self.get_input_type()
        
    def get_input_type(self):
        img_extension = ['.png', '.bmp', '.jpg', '.jpeg', 'tif']
        vid_extension = ['.mp4']
        if self.input_file.lower() == 'cam':
            self.input_type = 'cam'
        elif os.path.splitext(self.input_file)[-1] in img_extension:
            self.input_type = 'image'
        elif os.path.splitext(self.input_file)[-1] in vid_extension:
            self.input_type = 'video'
        log.info('Detected input type: ' + self.input_type)

    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for _ in range(10):
                _, frame=self.cap.read()
            yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
        cv2.destroyAllWindows()


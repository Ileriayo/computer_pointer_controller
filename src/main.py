import os
import sys
import cv2
import logging as log
from argparse import ArgumentParser
from openvino.inference_engine import IECore

from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarks


m_fd = '../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
m_hpe = '../models/intel/'
m_ld = '../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
m_ge = '../models/intel/'
input_stream = '../bin/demo.mp4'

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m_fd", type=str, default=m_fd, help="Path to a trained model for face detection")
    parser.add_argument("-m_hpe", type=str, default=m_hpe, help="Path to a trained model for human pose estimation")
    parser.add_argument("-m_ld", type=str, default=m_ld, help="Path to a trained model for facial landmark detection")
    parser.add_argument("-m_ge", type=str, default=m_ge, help="Path to a trained model for gaze estimation")
    parser.add_argument("-i", type=str, default=input_stream, help="Path to image or video file")
    parser.add_argument("-cpu_ext", required=False, type=str, default=None, help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-d", type=str, default="CPU", help="Specify the target device to infer on: " "CPU, GPU, FPGA or MYRIAD is acceptable. Sample " "will look for a suitable plugin for device " "specified (CPU by default)")
    return parser

def pipeline(args):
    feed=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
    feed.load_data()

    FaceDetectionPipe = FaceDetection(args.m_fd, args.d, args.cpu_ext)
    FaceDetectionPipe.load_model()

    FacialLandmarksPipe = FacialLandmarks(args.m_ld, args.d, args.cpu_ext)
    FacialLandmarksPipe.load_model()

    for frame in feed.next_batch():
        face_detection_output = FaceDetectionPipe.predict(frame)
        FacialLandmarksPipe.predict(face_detection_output)
    feed.close()


def main():
    args = build_argparser().parse_args()
    pipeline(args)

if __name__ == '__main__':
    main()
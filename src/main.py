import os
import sys
import cv2
import logging
import time
from argparse import ArgumentParser
from openvino.inference_engine import IECore

from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarks
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from model_metrics import ModelMetrics

log = logging.getLogger(__name__)

m_fd = '../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
m_hpe = '../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
m_ld = '../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
m_ge = '../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'
input_stream = '../bin/demo.mp4'

def build_argparser():
    '''
    Parse command line arguments.
    :return: command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-m_fd', type=str, default=m_fd, help='Path to a trained model for face detection')
    parser.add_argument('-m_hpe', type=str, default=m_hpe, help='Path to a trained model for head pose estimation')
    parser.add_argument('-m_ld', type=str, default=m_ld, help='Path to a trained model for facial landmark detection')
    parser.add_argument('-m_ge', type=str, default=m_ge, help='Path to a trained model for gaze estimation')
    parser.add_argument('-i', type=str, default=input_stream, help='Path to image or video file')
    parser.add_argument('-cpu_ext', required=False, type=str, default=None, help='MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.')
    parser.add_argument('-d', type=str, default='CPU', help='Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)')
    return parser

def pipeline(args):
    feed=InputFeeder(args.i)
    feed.load_data()

    FaceDetectionPipe = FaceDetection(args.m_fd, args.d, args.cpu_ext)
    load_time = time.time()
    FaceDetectionPipe.load_model()
    load_time_fd = time.time() - load_time

    FacialLandmarksPipe = FacialLandmarks(args.m_ld, args.d, args.cpu_ext)
    load_time = time.time()
    FacialLandmarksPipe.load_model()
    load_time_ld = time.time() - load_time

    HeadPoseEstimationPipe = HeadPoseEstimation(args.m_hpe, args.d, args.cpu_ext)
    load_time = time.time()
    HeadPoseEstimationPipe.load_model()
    load_time_hpe = time.time() - load_time

    GazeEstimationPipe = GazeEstimation(args.m_ge, args.d, args.cpu_ext)
    load_time = time.time()
    GazeEstimationPipe.load_model()
    load_time_ge = time.time() - load_time

    log.info('Load time for face detection model: ' + str(load_time_fd))
    log.info('Load time for landmark detection model: ' + str(load_time_ld))
    log.info('Load time for head pose estimation model: ' + str(load_time_hpe))
    log.info('Load time for gaze estimation model: ' + str(load_time_ge))

    frame_count = 0
    for frame in feed.next_batch():
        if frame is None:
            break

        frame_count += 1
        inf_time = time.time()
        face_detection_output = FaceDetectionPipe.predict(frame)
        inf_time_fd = time.time() - inf_time

        inf_time = time.time()
        eye_l_image, eye_r_image = FacialLandmarksPipe.predict(face_detection_output)
        inf_time_ld = time.time() - inf_time

        inf_time = time.time()
        yaw, pitch, roll = HeadPoseEstimationPipe.predict(face_detection_output)
        inf_time_hpe = time.time() - inf_time

        inf_time = time.time()
        x_coord, y_coord = GazeEstimationPipe.predict(eye_l_image, eye_r_image, [yaw, pitch, roll])
        inf_time_ge = time.time() - inf_time

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
        cv2.imshow('Frame', cv2.resize(frame, (600, 400)))

        pointer = MouseController('high', 'fast')
        pointer.move(x_coord, y_coord)

        fps_fd = 1 / inf_time_fd
        fps_ld = 1 / inf_time_ld
        fps_hpe = 1 / inf_time_hpe
        fps_ge = 1 / inf_time_ge

        log.info('Inference time for face detection model: ' + str(inf_time_fd))
        log.info('Inference time for landmark detection model: ' + str(inf_time_ld))
        log.info('Inference time for head pose estimation model: ' + str(inf_time_hpe))
        log.info('Inference time for gaze estimation model: ' + str(inf_time_ge))

        log.info('FPS for face detection model: ' + str(fps_fd))
        log.info('FPS for landmark detection model: ' + str(fps_ld))
        log.info('FPS for head pose estimation model: ' + str(fps_hpe))
        log.info('FPS for gaze estimation model: ' + str(fps_ge))

        log.info ('Frames Count:' + str(frame_count))
    feed.close()

def main():
    logging.basicConfig(level = logging.INFO)
    args = build_argparser().parse_args()
    pipeline(args)

if __name__ == '__main__':
    main()
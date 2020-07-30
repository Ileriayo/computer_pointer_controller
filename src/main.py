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
from visualize import Visualizer

log = logging.getLogger(__name__)

model_precision_fd = 'FP32-INT1'

# model_precision = 'FP16'
# model_precision = 'FP16-INT8'
model_precision = 'FP32'

# Default path for models (for ease of testing in development)
m_fd = '../models/intel/face-detection-adas-binary-0001/' + model_precision_fd +'/face-detection-adas-binary-0001'
m_hpe = '../models/intel/head-pose-estimation-adas-0001/' + model_precision + '/head-pose-estimation-adas-0001'
m_ld = '../models/intel/landmarks-regression-retail-0009/' + model_precision + '/landmarks-regression-retail-0009'
m_ge = '../models/intel/gaze-estimation-adas-0002/' + model_precision + '/gaze-estimation-adas-0002'
input_stream = '../bin/demo.mp4'

def build_argparser():
    '''
    Parse command line arguments.
    :return: command line arguments
    '''
    parser = ArgumentParser(description = 'Computer Pointer Controller')
    required = parser.add_argument_group('required arguments')

    required.add_argument('-m_fd', type=str, required=True, default=m_fd, help='Path to a trained model for face detection')
    required.add_argument('-m_hpe', type=str, required=True, default=m_hpe, help='Path to a trained model for head pose estimation')
    required.add_argument('-m_ld', type=str, required=True, default=m_ld, help='Path to a trained model for facial landmark detection')
    required.add_argument('-m_ge', type=str, required=True, default=m_ge, help='Path to a trained model for gaze estimation')
    required.add_argument('-i', type=str, required=True, default=input_stream, help='Path to image or video file, otherwise specify \'cam\' for live feed')
    required.add_argument('-d', type=str, required=False, default='CPU', help='Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)')
    required.add_argument('-pt', type=float, required=False, default=0.65, help='Probablity threshold for face detection')
    required.add_argument('-v', type=str2bool, required=False, default=False, help='Visualization flag - set to no display by default, to set display frames, specify \'t\' or \'yes\' or \'true\' or \'1\'')
    parser.add_argument('-cpu_ext', type=str, required=False, default=None, help='MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.')
    return parser

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def pipeline(args):
    feed=InputFeeder(args.i)
    feed.load_data()

    FaceDetectionPipe = FaceDetection(args.m_fd, args.pt, args.d, args.cpu_ext)
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

    inf_time_fd = inf_time_ld = inf_time_hpe = inf_time_ge = frame_count = 0
    for frame in feed.next_batch():
        if frame is None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

        frame_count += 1
        inf_time = time.time()
        fd_img_output, fd_coords = FaceDetectionPipe.predict(frame)
        inf_time_fd = time.time() - inf_time

        if (fd_coords == []):
            log.info('No face detected')
        else:
            inf_time = time.time()
            eye_l_image, eye_r_image, ld_coords = FacialLandmarksPipe.predict(fd_img_output)
            inf_time_ld = time.time() - inf_time

            inf_time = time.time()
            hpe_output = HeadPoseEstimationPipe.predict(fd_img_output)
            inf_time_hpe = time.time() - inf_time

            yaw, pitch, roll = hpe_output
            inf_time = time.time()
            ge_output = GazeEstimationPipe.predict(eye_l_image, eye_r_image, [yaw, pitch, roll])
            inf_time_ge = time.time() - inf_time

            if frame_count % 5 == 0:
                pointer = MouseController('medium', 'fast')
                pointer.move(ge_output[0], ge_output[1])

            fps_fd = 1 / inf_time_fd
            fps_ld = 1 / inf_time_ld
            fps_hpe = 1 / inf_time_hpe
            fps_ge = 1 / inf_time_ge
        
            if (args.v):
                v = Visualizer(frame, fd_img_output, fd_coords, ld_coords, hpe_output)
                v.visualize()

            log.info('Average inference time for face detection model: ' + str(inf_time_fd))
            log.info('Average inference time for landmark detection model: ' + str(inf_time_ld))
            log.info('Average inference time for head pose estimation model: ' + str(inf_time_hpe))
            log.info('Average inference time for gaze estimation model: ' + str(inf_time_ge))

            log.info('FPS for face detection model: ' + str(fps_fd))
            log.info('FPS for landmark detection model: ' + str(fps_ld))
            log.info('FPS for head pose estimation model: ' + str(fps_hpe))
            log.info('FPS for gaze estimation model: ' + str(fps_ge))

            log.info ('Frames Count:' + str(frame_count))

            mm = ModelMetrics()
            log.info('Writing stats to file...')
            mm.save_to_file('stats_fd.txt', 'FD/' + model_precision, inf_time_fd, fps_fd, load_time_fd)
            mm.save_to_file('stats_ld.txt', model_precision, inf_time_ld, fps_ld, load_time_ld)
            mm.save_to_file('stats_hpe.txt', model_precision, inf_time_hpe, fps_hpe, load_time_hpe)
            mm.save_to_file('stats_ge.txt', model_precision, inf_time_ge, fps_ge, load_time_ge)
    feed.close()

def main():
    logging.basicConfig(level = logging.INFO)
    args = build_argparser().parse_args()
    pipeline(args)

if __name__ == '__main__':
    main()
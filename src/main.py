import os
import sys
import cv2
import logging as log
from argparse import ArgumentParser
from openvino.inference_engine import IECore

from input_feeder import InputFeeder
from facial_landmarks_detection import FacialLandmarks


m_df = '../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
m_hpe = '../models/intel/'
m_ld = '../models/intel/'
m_ge = '../models/intel/'
input_stream = '../bin/demo.mp4'

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m_fd", type=str, default=m_df, help="Path to an xml file with a trained model.")
    parser.add_argument("-m_hpe", type=str, default=m_hpe, help="Path to an xml file with a trained model.")
    parser.add_argument("-m_ld", type=str, default=m_ld, help="Path to an xml file with a trained model.")
    parser.add_argument("-m_ge", type=str, default=m_ge, help="Path to an xml file with a trained model.")
    parser.add_argument("-i", type=str, default=input_stream, help="Path to image or video file")
    parser.add_argument("-cpu_ext", required=False, type=str, default=None, help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-d", type=str, default="CPU", help="Specify the target device to infer on: " "CPU, GPU, FPGA or MYRIAD is acceptable. Sample " "will look for a suitable plugin for device " "specified (CPU by default)")
    parser.add_argument("-pt", type=float, default=0.5, help="Probability threshold for detections filtering" "(0.5 by default)")
    return parser

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_bin = model_name + '.bin'
        self.model_xml = model_name + '.xml'
        self.device = device
        self.cpu_extension = extensions

        try:
            self.ie_plugin = IECore()
            self.model=self.ie_plugin.read_network(model=self.model_xml, weights=self.model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        # self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Add CPU extension to IECore
        if self.cpu_extension and 'CPU' in self.device:
            log.info('Adding CPU extension:\n\t{}'.format(self.cpu_extension))
            self.ie_plugin.add_extension(self.cpu_extension, self.device)

        # Check layers
        log.info('Current device specified: {}'.format(self.device))
        log.info("Checking for unsupported layers...")
        supported_layers = self.ie_plugin.query_network(network=self.model, device_name=self.cpu_extension)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error('These layers are unsupported:\n{}'.format(', '.join(unsupported_layers)))
            log.error('Specify an available extension to add to IECore from the command line using -l/--cpu_extension')
            exit(1)
        else:
            log.info('All layers are supported!')

        # Load the model network into IECore
        self.exec_network = self.ie_plugin.load_network(self.model, self.device)
        log.info("IR Model has been successfully loaded into IECore")

        return self.exec_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_image = self.preprocess_input(image)
        self.exec_network.start_async(0, {self.input_name: p_image})
        
        if self.wait() == 0:
            outputs = self.get_outputs()
            coords = self.preprocess_output(outputs)            
            # return self.draw_output(coords, image)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        required_width = self.input_shape[2]
        required_height = self.input_shape[3]
        dimension = (required_height, required_width)

        image = cv2.resize(image, dimension)
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_outputs(self):
        return self.exec_network.requests[0].outputs[self.output_name]

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        left_x = outputs[0][0] * image.shape[1]
        left_y = outputs[0][1] * image.shape[0]
        right_x = outputs[0][2] * image.shape[1]
        right_y = outputs[0][3] * image.shape[0]

        return {
            'left_eye': [left_x, left_y],
            'right_eye': [right_x, right_y]
        }


def pipeline(args):
    feed=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
    feed.load_data()

    FacialLandmarksPipe = FacialLandmarks(args.m_ld, args.d, args.cpu_ext)
    FacialLandmarksPipe.load_model()

    for frame in feed.next_batch():
        FacialLandmarksPipe.predict(frame)
    feed.close()


def main():
    args = build_argparser().parse_args()
    pipeline(args)

if __name__ == '__main__':
    main()
import os
import sys
import cv2
import logging
from openvino.inference_engine import IECore

log = logging.getLogger(__name__)

class FacialLandmarks:
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
        except Exception:
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
        supported_layers = self.ie_plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error('These layers are unsupported:\n{}'.format(', '.join(unsupported_layers)))
            log.error('Specify an available extension to add to IECore from the command line using "-l"')
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
            left_eye, right_eye = self.preprocess_output(outputs, image)
        # cv2.imwrite('../images/outputs/left_eye.jpg', left_eye)
        # cv2.imwrite('../images/outputs/right_eye.jpg', right_eye)
        return left_eye, right_eye

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

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # denormalize detections
        xl = int(outputs[0][0][0] * image.shape[1])
        yl = int(outputs[0][1][0] * image.shape[0])
        xr = int(outputs[0][2][0] * image.shape[1])
        yr = int(outputs[0][3][0] * image.shape[0])

        # include offset for left eye
        xlmin = xl - 15
        ylmin = yl - 15
        xlmax = xl + 15
        ylmax = yl + 15

        # include offset for right eye
        xrmin = xr - 15
        yrmin = yr - 15
        xrmax = xr + 15
        yrmax = yr + 15

        # draw boxes around eyes
        # cv2.rectangle(image, (xlmin, ylmin), (xlmax, ylmax), (0, 0, 255), 1)
        # cv2.rectangle(image, (xrmin, yrmin), (xrmax, yrmax), (0, 0, 255), 1)

        # crop eyes
        eye_l = image[ylmin:ylmax, xlmin:xlmax]
        eye_r = image[yrmin:yrmax, xrmin:xrmax]

        return eye_l, eye_r

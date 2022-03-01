import logging
import time

import numpy as np

from autolab_core import ColorImage, DepthImage, RgbdImage,CameraIntrinsics
from .native_phoxi_driver import NativePhoXiDriver


logger = logging.getLogger('phoxipy.PhoXiSensor')


class SensorIOException(Exception):
    pass


class PhoXiSensor:
    FOCAL_LENGTH = 1105.0 / 1032.0 # this is the normalized focal length, to use it multiply by the image WIDTH to get the pixel value

    def __init__(self, device_name: str, inpaint=False, crop=(0, 0, 0, 0)):
        self._device_name = str(device_name)
        self._native_driver = NativePhoXiDriver(self._device_name)
        self._is_running = False

        # Set up camera intrinsics for the sensor
        #these are the default, but user should check the image size and 
        #appropriately set them if need be
        self._camera_intr = self.create_intr(2064, 1544)

        self._crop = crop
        self._inpaint = inpaint
        if np.any(self._crop):
            self._camera_intr = self._camera_intr.crop(*self._crop)

    @staticmethod
    def create_intr(width: int, height: int):
        #not a bug to have focal_length*width for both, that is intended
        return CameraIntrinsics(
            fx=PhoXiSensor.FOCAL_LENGTH*width,
            fy=PhoXiSensor.FOCAL_LENGTH*width,
            cx=width/2.0, cy=height/2.0,
            width=width, height=height,
            frame='phoxi'
        )

    @property
    def device_name(self):
        return self._device_name

    @property
    def is_running(self):
        return self._is_running

    @property
    def intrinsics(self):
        return self._camera_intr
    
    @intrinsics.setter
    def intrinsics(self,new_in):
        self._camera_intr=new_in
    
    @property
    def ir_intrinsics(self):
        return self._camera_intr
    
    @property
    def ir_frame(self):
        return self.intrinsics.frame

    @property 
    def frame(self):
        return self.intrinsics.frame

    def frames(self):
        #need this to work with CameraChessboardRegistration
        image = self.read() #rgbdimage
        return image.color, image.depth, None

    def read(self):
        """Read data from the sensor and return it.

        Returns
        -------
        data : :class:`.GDImage`
            The grayscale-depth image from the sensor.
        """
        self._color_im = None
        self._depth_im = None

        try:
            start = time.time()
            logger.debug('Reading frames from Photoneo PhoXi {}...'.format(self.device_name))
            color, depth = self._native_driver.read()
            logger.debug('Reading from Photoneo PhoXi {} took {:.3}s'.format(self.device_name, time.time() - start))
        except ValueError:
            logger.error('PhoXi sensor disconnected, check PhoXiControl')
            raise SensorIOException('PhoXi sensor not connected')

        # color = color* 255.0 / 4096.0
        # scaling changed by justin on july 19
        # heuristic for scaling RAW to normal image is taken from 
        # https://www.cs.cmu.edu/~16385/lectures/lecture17.pdf
        # slide 69
        scaled_color = np.power(color, 1.0/2.2)
        color = scaled_color * 255.0 / np.max(scaled_color)
        color = np.repeat(color[...,None], 3, axis=-1).astype(np.uint8)
        depth = depth / 1000.0

        phoxi_depth_im = DepthImage(depth, frame=self.frame)
        phoxi_color_im = ColorImage(color, frame=self.frame)

        if np.any(self._crop):
            phoxi_color_im = phoxi_color_im.crop(*self._crop)
            phoxi_depth_im = phoxi_depth_im.crop(*self._crop)
        
        if self._inpaint:
            phoxi_color_im = phoxi_color_im.inpaint()
            phoxi_depth_im = phoxi_depth_im.inpaint()

        return RgbdImage.from_color_and_depth(phoxi_color_im, phoxi_depth_im)

    def read_orthographic(self):
        """reads both the normal (projective) as well as orthographic image"""
        color, depth, ortho_color, ortho_depth = self._native_driver.read_orthographic()

        # Normal
        scaled_color = np.power(color, 1.0/2.2)
        color = scaled_color * 255.0 / np.max(scaled_color)
        color = np.repeat(color[...,None], 3, axis=-1).astype(np.uint8)
        depth = np.array(depth) / 1000.0

        phoxi_color_im = ColorImage(color, frame=self.frame)
        phoxi_depth_im = DepthImage(depth, frame=self.frame)
        normal = RgbdImage.from_color_and_depth(phoxi_color_im, phoxi_depth_im)

        # Orthographic
        scaled_color = np.array(ortho_color)[:,:,0]
        color = scaled_color * 255.0 / np.max(scaled_color)
        color = np.repeat(color[..., None], 3, axis=-1).astype(np.uint8)
        depth = (850.0 + np.array(ortho_depth) / (256 * 256 - 1) * (1250.0 - 850.0)) / 1000.0

        phoxi_color_im = ColorImage(color, frame=self.frame)
        phoxi_depth_im = DepthImage(depth, frame=self.frame)
        orthographic = RgbdImage.from_color_and_depth(phoxi_color_im, phoxi_depth_im)

        return normal, orthographic

    def start(self):
        if self.is_running:
            logger.warning('PhoXi is already started')
            return True
        
        if not self._native_driver.start():
            logger.warning('Failed to start Photoneo PhoXi')
            self._is_running = False

        logger.info('Started Photoneo PhoXi {}'.format(self.device_name))
        self._is_running = True
        return self._is_running

    def stop(self):
        if not self._is_running:
            logger.warning('PhoXi not running, cannot stop')
            return False
        
        self._native_driver.stop()
        logger.info('Stopped Photoneo PhoXi {}'.format(self.device_name))
        self._is_running = False
        return True

    def __del__(self):
        if self._is_running:
            self.stop()

#!/bin/env/python3
"""
RTSP client for getting video stream from SIYI cameras, e.g. ZR10, A8 Mini
ZR10 webpage: http://en.siyi.biz/en/Gimbal%20Camera/ZR10/overview/
A8 mini page: https://shop.siyi.biz/products/siyi-a8-mini?VariantsId=10611
Author : Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Copyright 2022

Required:
- OpenCV
    (sudo apt-get install python3-opencv -y)
- imutils
    pip install imutils
- Gstreamer (https://gstreamer.freedesktop.org/documentation/installing/index.html?gi-language=c)
    (Ubuntu: sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio -y)
- Deepstream (only for Nvidia Jetson boards)
    (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#jetson-setup)
- For RTMP streaming
    sudo apt install ffmpeg -y
    pip install ffmpeg-python
"""
import os
import cv2
import logging
import subprocess
from time import time, sleep
import threading

class SIYIRTSP:
    def __init__(
        self,
        rtsp_url="rtsp://192.168.144.25:8554/main.264",
        cam_name="ZR10",
        debug=False,
        use_udp=True,
        allow_tcp_fallback=True,
        low_latency=True,
    ) -> None:
        '''
        Receives video stream from SIYI cameras

        Params
        --
        - rtsp_url [str] RTSP url
        - cam_name [str] camera name (optional)
        - debug [bool] print debug messages
        - use_udp [bool] use UDP instead of TCP for RTSP transport
        - allow_tcp_fallback [bool] retry with default transport if UDP open fails
        - low_latency [bool] reduce decoder buffering for live preview
        '''
        self._original_rtsp_url = rtsp_url  # Keep the original URL intact
        self._rtsp_url = rtsp_url
        self._cam_name = cam_name
        self._use_udp = use_udp  # Track whether we are trying UDP or TCP
        self._allow_tcp_fallback = allow_tcp_fallback
        self._low_latency = low_latency
        self._last_error = ""
        self._opened = False
        self._backend = "unopened"
        self._capture_options = ""

        # Desired image width/height
        self._width = 640  # Lower resolution to reduce data size
        self._height = 480

        # Stored image frame
        self._frame = None
        self._frame_lock = threading.Lock()

        # Configure logging
        self._debug = debug
        self._logger = logging.getLogger(self.__class__.__name__)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s [SIYIRTSP::%(funcName)s]: %(message)s'))
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging.DEBUG if self._debug else logging.INFO)
        self._logger.propagate = False

        # Flag to stop frame grabbing loop and close windows
        self._stopped = False
        self._recv_thread = None  # Initialize thread variable
        self._stream = None

        # Show grabbed frame in a window (mainly for debugging)
        self._show_window = False

        self._last_image_time = time()

        # Timeout (seconds) before closing everything
        self._connection_timeout = 2.0

        # Start stream
        self.start()

    def setShowWindow(self, f: bool):
        self._show_window = f

    def getFrame(self):
        """
        Returns current image frame
        """
        with self._frame_lock:
            return self._frame

    def isOpened(self):
        return self._opened

    def getLastError(self):
        return self._last_error

    def getBackend(self):
        return self._backend

    def _build_ffmpeg_capture_options(self):
        transport = "udp" if self._use_udp else "tcp"
        options = [
            ("rtsp_transport", transport),
        ]

        if self._low_latency:
            options.extend([
                ("fflags", "nobuffer"),
                ("flags", "low_delay"),
                ("avioflags", "direct"),
                ("probesize", "32"),
                ("analyzeduration", "0"),
                ("reorder_queue_size", "0"),
                ("buffer_size", "65536"),
                ("max_delay", "500000"),
            ])

        return "|".join(f"{key};{value}" for key, value in options)

    def _open_ffmpeg_stream(self):
        self._capture_options = self._build_ffmpeg_capture_options()
        previous_capture_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")

        if self._capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._capture_options
            self._logger.debug("Using FFmpeg capture options: %s", self._capture_options)

        try:
            stream = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
        finally:
            if previous_capture_options is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_capture_options

        self._backend = "ffmpeg"
        return stream

    def start(self):
        """
        Start receiving thread
        """
        try:
            self._logger.info("Connecting to %s using %s...", self._cam_name, "UDP" if self._use_udp else "TCP")
            self._stopped = False
            self._opened = False
            self._last_error = ""
            self._backend = "unopened"

            # Initialize the FFmpeg-based VideoCapture
            self._stream = self._open_ffmpeg_stream()

            # Reduce buffer size for lower latency
            self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._low_latency:
                # These properties are best-effort on RTSP sources and can add work
                # on some OpenCV builds, so they stay disabled for low-latency mode.
                self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._stream.set(cv2.CAP_PROP_FPS, 15)

            if not self._stream.isOpened():
                raise Exception(f"Failed to open RTSP stream {self._rtsp_url}")

            # Start the receiving loop thread
            self._recv_thread = threading.Thread(target=self.loop, daemon=True)
            self._recv_thread.start()
            self._opened = True
            return True
        except Exception as e:
            self._last_error = str(e)
            self._logger.error("Could not receive stream from %s. Error: %s", self._cam_name, e)

            if self._stream:
                self._stream.release()
                self._stream = None

            if self._use_udp and self._allow_tcp_fallback:
                self._logger.info("Retrying with default RTSP transport...")
                self._use_udp = False
                return self.start()

            self.close()
            return False

    def close(self):
        self._logger.info("Closing stream of %s...", self._cam_name)
        self._stopped = True
        self._opened = False
        cv2.destroyAllWindows()
        if self._stream:
            self._stream.release()
            self._stream = None
        if self._recv_thread and self._recv_thread.is_alive() and self._recv_thread is not threading.current_thread():
            self._recv_thread.join()

    def loop(self):
        self._last_image_time = time()

        while not self._stopped:
            ret, frame = self._stream.read()

            if not ret:
                if (time() - self._last_image_time) > self._connection_timeout:
                    self._logger.warning("Connection timeout. Exiting")
                    self.close()
                    break
                continue

            self._last_image_time = time()
            with self._frame_lock:
                self._frame = frame

            # Log the timestamp
            timestamp = self._stream.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp == 0:
                timestamp = time() * 1000  # Convert seconds to milliseconds for consistency
            self._logger.debug(f"Frame timestamp: {timestamp} ms")

            if self._show_window:
                cv2.imshow('{} Stream'.format(self._cam_name), frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.close()
                    break

            # Optimized delay to avoid overwhelming CPU while reducing latency
            sleep(0.001)

        self._logger.warning("RTSP receiving loop is done")
        return

class RTMPSender:
    '''
    Streams image frames to an RTMP server
    '''
    def __init__(self, rtmp_url="rtmp://127.0.0.1:1935/live/webcam", debug=False) -> None:
        '''
        Params
        --
        - rtmp_url [str] RTMP server URL
        - debug [bool] Printing debug message
        '''
        self._rtmp_url=rtmp_url
        # Desired frequency of streaming to rtmp server
        self._fps =30

        self._last_send_time = time()
        self._current_send_time = time()


        # Frame to send
        self._frame = None

        # Desired image height
        self._height = 480
        # Desired image width
        self._width = 640

        self._toGray=False
        if self._toGray:
            self._pix_fmt="gray"
        else:
            self._pix_fmt="bgr24"

        # Flag to stop streaming loop
        self._stopped = False

        self._debug= debug # print debug messages
        if self._debug:
            d_level = logging.DEBUG
        else:
            d_level = logging.INFO
        LOG_FORMAT=' [%(levelname)s] %(asctime)s [RTMPSender::%(funcName)s] :\t%(message)s'
        logging.basicConfig(format=LOG_FORMAT, level=d_level)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.propagate = False

        # Streaming thread
        self._st_thread = threading.Thread(target=self.loop)
        self._p = None

        
    def setImageSize(self, w=640, h=480):
        self._width=w
        self._height=h

    def setFPS(self, fps=20):
        self._fps=fps

    def setGrayFrame(self, b: bool):
        '''
        Params
        --
        - b [bool] True: sends grayscale image. Otherwise sends color image
        '''
        self._toGray = b
        if self._toGray:
            self._pix_fmt="gray"
        else:
            self._pix_fmt="bgr24"


    def setFrame(self, frame):
        self._frame=frame


    def start(self):
        command = ["ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", self._pix_fmt,
            "-s", "{}x{}".format(self._width, self._height),
            "-r", str(self._fps),
            "-i", "-",
            "-c:v", "libx264",
            '-pix_fmt', 'yuv420p',
            "-preset", "ultrafast",
            "-f", "flv",
            "-tune", "zerolatency",
            self._rtmp_url]
        # using subprocess and pipe to fetch frame data
        try:
            self._p = subprocess.Popen(command, stdin=subprocess.PIPE)
        except Exception as e:
            self._logger.error("Could not create ffmpeg pipeline. Error %s", e)
            self._p = None
            return False

        try:
            self._st_thread.start()
        except Exception as e:
            self._logger.error("Could not start RTMP streaming thread. Error %s", e)
            if self._p is not None:
                self._p.kill()
                self._p = None
            return False

        return True

    def stop(self):
        """
        Stops streaming thread
        """
        self._logger.warning("RTMP streaming is stopped.")
        self._stopped=True
        if self._p is not None:
            try:
                if self._p.stdin:
                    self._p.stdin.close()
            except Exception:
                pass
            try:
                self._p.kill()
            except Exception:
                pass

    def sendFrame(self) -> bool:
        '''
        Sends current image frame stored in self._frame

        Returns
        --
        True if all is fine. False otherwise
        '''
        if self._frame is None:
            return False

        try:
            val = self._frame.shape
            rows = val[0]
            cols=val[1]
            # Resize the image, if needed
            if rows != self._height or cols != self._width:
                self._frame = cv2.resize(self._frame, (self._width,self._height), interpolation = cv2.INTER_AREA)

            if self._toGray and (len(self._frame.shape) > 2):
                self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

            self._p.stdin.write(self._frame.tobytes())

            return(True)
        except Exception as e:
            self._logger.error(" Error in sending:  %s", e)
            return(False)

    def loop(self):
        while(not self._stopped):
            dt = time() - self._last_send_time
            start_t=time()
            self.sendFrame()
            end_t=time()
            dt=end_t-start_t
            duration = 1/self._fps
            if  dt < duration:
                sleep(duration-dt)

        self._logger.warning("RTMP streaming loop is done")
        return

        


def test():
    # rtsp = SIYIRTSP(debug=False)
    # rtsp.setShowWindow(True)
    # Webcam
    try:
        wc = cv2.VideoCapture(src=0)
        wc.open(0)
    except Exception as e:
        print("Error in opening webcam")
        exit(1)

    rtmp = RTMPSender(rtmp_url="rtmp://127.0.0.1:1935/live/webcam")
    rtmp.start()
    try:
        while(True):
            # frame=stream.getFrame()
            frame=wc.read()
            rtmp.setFrame(frame)
    except KeyboardInterrupt:
        # rtsp.close()
        rtmp.stop()
        wc.stop()
        cv2.destroyAllWindows()
        # quit
        exit(0)

    

if __name__=="__main__":
    test()

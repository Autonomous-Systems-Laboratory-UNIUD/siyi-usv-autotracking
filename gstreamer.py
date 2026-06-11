import logging
import threading

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GStreamerFrameStreamer:
    """Streamer GStreamer based on UDP RTP H264 which keeps in memory always the last frame."""

    FRAME_TIMEOUT_NS = Gst.SECOND // 20  # ~50 FPS max

    def __init__(self, port: int, logger=None):
        self.port = port
        self.sink_name = "sink"
        self.logger = logger or logging.getLogger(__name__)

        self._latest_frame = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        self._frame_shape = None
        self._frame_size_logged = False


    def build_pipeline(self, max_buffers: int = 1) -> str:
        return f"""
            udpsrc port={self.port}
                caps="application/x-rtp,media=video,encoding-name=H264,payload=96" !
            rtpjitterbuffer !
            rtph264depay !
            h264parse !
            avdec_h264 !
            videoconvert !
            video/x-raw,format=BGR !
            appsink
                name={self.sink_name}
                emit-signals=false
                max-buffers={max_buffers}
                drop=true
                sync=false
        """


    def start(self):
        if self._thread and self._thread.is_alive():
            return

        Gst.init(None)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()


    def stop(self, timeout: float = 1.0):
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout)
            self._thread = None

        with self._latest_lock:
            self._latest_frame = None


    def get_frame(self, copy: bool = True):
        with self._latest_lock:
            frame = self._latest_frame

        if frame is None:
            return None

        return frame.copy() if copy else frame


    def _run(self):
        pipeline = None

        try:
            pipeline = Gst.parse_launch(self.build_pipeline())
            appsink = pipeline.get_by_name(self.sink_name)

            if appsink is None:
                raise RuntimeError(f"Appsink '{self.sink_name}' not found")

            bus = pipeline.get_bus()

            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")

            self.logger.info("GStreamer pipeline started on port %s", self.port)

            while not self._stop_event.is_set():

                # Handle error/EOS
                message = bus.timed_pop_filtered(
                    0,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS,
                )

                if message and not self._handle_bus_message(message):
                    break

                # Pull frame
                sample = appsink.emit("try-pull-sample", self.FRAME_TIMEOUT_NS)

                if sample is None:
                    continue

                frame = self._sample_to_numpy(sample)

                if frame is not None:
                    with self._latest_lock:
                        self._latest_frame = frame

        except Exception:
            self.logger.exception("GStreamer streamer crashed")

        finally:
            if pipeline:
                pipeline.set_state(Gst.State.NULL)


    def _sample_to_numpy(self, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        try:
            width = structure.get_value("width")
            height = structure.get_value("height")

            if self._frame_shape is None:
                self._frame_shape = (height, width, 3)
                self.logger.info("Frame size: %dx%d", width, height)

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map GStreamer buffer")
                return None

            try:
                frame = np.frombuffer(map_info.data, dtype=np.uint8)
                return frame.reshape(self._frame_shape).copy()
            finally:
                buf.unmap(map_info)

        except Exception:
            self.logger.exception("Failed to convert sample to numpy frame")
            return None


    def _handle_bus_message(self, message) -> bool:
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(
                "GStreamer error from %s: %s",
                message.src.get_name() if message.src else "unknown",
                err,
            )
            if debug:
                self.logger.debug("Debug: %s", debug)
            return False

        if msg_type == Gst.MessageType.EOS:
            self.logger.info("GStreamer end of stream")
            return False

        return True
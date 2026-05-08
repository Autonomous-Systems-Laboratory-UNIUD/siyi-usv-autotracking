import logging
import threading

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GStreamerFrameStreamer:
    """Read the latest frame from an RTSP GStreamer pipeline in a background thread."""

    def __init__(
        self,
        rtsp_url,
        rtsp_transport="udp",
        rtsp_latency=10,
        sink_name="sink",
        logger=None,
    ):
        self.rtsp_url = rtsp_url
        self.rtsp_transport = rtsp_transport
        self.rtsp_latency = rtsp_latency
        self.sink_name = sink_name
        self.logger = logger or logging.getLogger(__name__)

        self._latest_frame = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._frame_size_logged = False

    def build_pipeline(self):
        return f"""
rtspsrc location={self.rtsp_url} latency={self.rtsp_latency} protocols={self.rtsp_transport} !
rtph264depay !
h264parse !
avdec_h264 !
videoconvert !
video/x-raw,format=BGR !
appsink name={self.sink_name} max-buffers=1 drop=false sync=true
"""

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        Gst.init(None)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=1.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_frame(self, copy=True):
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            if copy:
                return self._latest_frame.copy()
            return self._latest_frame

    def _run(self):
        pipeline = None

        try:
            pipeline = Gst.parse_launch(self.build_pipeline())
            appsink = pipeline.get_by_name(self.sink_name)
            if appsink is None:
                raise RuntimeError(f"Could not find appsink named '{self.sink_name}'")

            bus = pipeline.get_bus()
            state_change = pipeline.set_state(Gst.State.PLAYING)
            if state_change == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set GStreamer pipeline to PLAYING")

            while not self._stop_event.is_set():
                message = bus.timed_pop_filtered(
                    0,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS,
                )
                if message is not None and not self._handle_bus_message(message):
                    self._stop_event.set()
                    break

                sample = appsink.emit("try-pull-sample", Gst.SECOND // 20)
                if sample is None:
                    continue

                frame = self._sample_to_numpy(sample)
                if frame is not None:
                    with self._latest_lock:
                        self._latest_frame = frame
        except Exception as exc:
            self.logger.error("GStreamer streamer stopped: %s", exc)
            self._stop_event.set()
        finally:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)

    def _sample_to_numpy(self, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        if not self._frame_size_logged:
            self.logger.info("Frame size: %sx%s", width, height)
            self._frame_size_logged = True

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            self.logger.error("Failed to map buffer")
            return None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            return frame.reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

    def _handle_bus_message(self, message):
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(
                "GStreamer error from %s: %s",
                message.src.get_name(),
                err,
            )
            if debug:
                self.logger.error("Debug details: %s", debug)
            return False

        if msg_type == Gst.MessageType.EOS:
            self.logger.info("GStreamer end of stream")
            return False

        return True

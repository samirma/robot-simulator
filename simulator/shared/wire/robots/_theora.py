"""theora_image_transport's publisher (1.15.0, theora_publisher.cpp) in Python, through
libtheora's own encoder (libtheoraenc, ctypes): the same encoder setup, colour conversion,
header packets and packet fields, so a wire that serves image_transport's theora plugin
publishes what the C++ plugin would.

Setup (theora_publisher.cpp L48-L65): pic offset 0, colorspace unspecified, 4:2:0, aspect
1/1, fps 1/1, keyframe_granule_shift 6; quality / target bitrate from the plugin's
dynamic_reconfigure config (configCb L91-L138: target_bitrate 0 unless optimize_for is
Bitrate). The context is made on the first frame (ensureEncodingContext L246-L285): frame
size rounded up to multiples of 16, the keyframe frequency forced, the header packets
flushed (comment vendor "Willow Garage theora_image_transport"), kept for new subscribers
(connectCallback L140-L146) and published. Each frame (publish L156-L239): to bgr8,
padded, BGR -> YCrCb, Y as is, Cr and Cb pyrDown'ed, submitted, and every packet out
published with the image's header (oggPacketToMsg L287-L297)."""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np

try:  # OpenCV before libtheora: on aarch64 its libgomp needs static TLS left
    import cv2  # noqa: F401
except ImportError:  # the host's unit tests fake this module's encoder
    cv2 = None

TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE = 4
TH_ENCCTL_SET_QUALITY = 28
TH_ENCCTL_SET_BITRATE = 30
TH_EINVAL = -10
VENDOR = b"Willow Garage theora_image_transport"


class ThInfo(ctypes.Structure):
    _fields_ = [("version_major", ctypes.c_ubyte), ("version_minor", ctypes.c_ubyte),
                ("version_subminor", ctypes.c_ubyte),
                ("frame_width", ctypes.c_uint32), ("frame_height", ctypes.c_uint32),
                ("pic_width", ctypes.c_uint32), ("pic_height", ctypes.c_uint32),
                ("pic_x", ctypes.c_uint32), ("pic_y", ctypes.c_uint32),
                ("fps_numerator", ctypes.c_uint32), ("fps_denominator", ctypes.c_uint32),
                ("aspect_numerator", ctypes.c_uint32), ("aspect_denominator", ctypes.c_uint32),
                ("colorspace", ctypes.c_int), ("pixel_fmt", ctypes.c_int),
                ("target_bitrate", ctypes.c_int), ("quality", ctypes.c_int),
                ("keyframe_granule_shift", ctypes.c_int)]


class ThComment(ctypes.Structure):
    _fields_ = [("user_comments", ctypes.POINTER(ctypes.c_char_p)),
                ("comment_lengths", ctypes.POINTER(ctypes.c_int)),
                ("comments", ctypes.c_int), ("vendor", ctypes.c_char_p)]


class ThImgPlane(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("stride", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_ubyte))]


class OggPacket(ctypes.Structure):
    _fields_ = [("packet", ctypes.POINTER(ctypes.c_ubyte)), ("bytes", ctypes.c_long),
                ("b_o_s", ctypes.c_long), ("e_o_s", ctypes.c_long),
                ("granulepos", ctypes.c_int64), ("packetno", ctypes.c_int64)]


def _lib():
    # libtheoraenc needs libtheoradec's symbols (th_info_init, th_comment_*): loaded first,
    # globally
    dec = ctypes.CDLL(ctypes.util.find_library("theoradec") or "libtheoradec.so.1",
                      mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(ctypes.util.find_library("theoraenc") or "libtheoraenc.so.1")
    lib.th_info_init = dec.th_info_init
    lib.th_comment_init = dec.th_comment_init
    lib.th_encode_alloc.restype = ctypes.c_void_p
    lib.th_encode_alloc.argtypes = [ctypes.POINTER(ThInfo)]
    lib.th_encode_free.argtypes = [ctypes.c_void_p]
    lib.th_encode_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.th_encode_flushheader.argtypes = [ctypes.c_void_p, ctypes.POINTER(ThComment),
                                          ctypes.POINTER(OggPacket)]
    lib.th_encode_ycbcr_in.argtypes = [ctypes.c_void_p, ThImgPlane * 3]
    lib.th_encode_packetout.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(OggPacket)]
    lib.th_info_init.argtypes = [ctypes.POINTER(ThInfo)]
    lib.th_comment_init.argtypes = [ctypes.POINTER(ThComment)]
    return lib


class TheoraPublisher:
    """One theora stream: `configure(config)` with the plugin's dynamic_reconfigure values,
    `encode(header, rgb, encoding)` -> the packets to publish ([(fields), ...]: header
    packets first when the context is new), `stream_header` the header packets for a new
    subscriber. A packet is a dict of Packet fields (header, data, b_o_s, e_o_s,
    granulepos, packetno)."""

    def __init__(self):
        self.lib = _lib()
        self.setup = ThInfo()
        self.lib.th_info_init(ctypes.byref(self.setup))
        s = self.setup
        s.pic_x = s.pic_y = 0
        s.colorspace = 0          # TH_CS_UNSPECIFIED
        s.pixel_fmt = 0           # TH_PF_420
        s.aspect_numerator = s.aspect_denominator = 1
        s.fps_numerator = s.fps_denominator = 1
        s.keyframe_granule_shift = 6
        s.target_bitrate = -1
        s.quality = -1
        self.keyframe_frequency = 64
        self.ctx = None
        self.stream_header = []

    def _free(self):
        if self.ctx:
            self.lib.th_encode_free(self.ctx)
        self.ctx = None

    def configure(self, config):
        bitrate = int(config["target_bitrate"]) if int(config["optimize_for"]) == 0 else 0
        s = self.setup
        update_bitrate = bool(bitrate) and s.target_bitrate != bitrate
        update_quality = not bitrate and (s.quality != int(config["quality"]) or s.target_bitrate > 0)
        s.quality = int(config["quality"])
        s.target_bitrate = bitrate
        self.keyframe_frequency = int(config["keyframe_frequency"])
        if self.ctx:
            err = 0
            if update_bitrate:
                v = ctypes.c_long(bitrate)
                err = self.lib.th_encode_ctl(self.ctx, TH_ENCCTL_SET_BITRATE, ctypes.byref(v),
                                             ctypes.sizeof(v))
            if update_quality:
                v = ctypes.c_int(int(config["quality"]))
                err = self.lib.th_encode_ctl(self.ctx, TH_ENCCTL_SET_QUALITY, ctypes.byref(v),
                                             ctypes.sizeof(v))
            if err:   # unable to change it on the fly: a new context (and new headers)
                self._free()
            else:
                self._keyframe_frequency()
                config["keyframe_frequency"] = self.keyframe_frequency
        return config

    def _keyframe_frequency(self):
        v = ctypes.c_uint32(self.keyframe_frequency)
        self.lib.th_encode_ctl(self.ctx, TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE, ctypes.byref(v),
                               ctypes.sizeof(v))
        self.keyframe_frequency = int(v.value)

    @staticmethod
    def _packet(header, op):
        return {"header": header, "data": ctypes.string_at(op.packet, op.bytes),
                "b_o_s": int(op.b_o_s), "e_o_s": int(op.e_o_s),
                "granulepos": int(op.granulepos), "packetno": int(op.packetno)}

    def _ensure(self, header, width, height):
        s = self.setup
        if self.ctx and s.pic_width == width and s.pic_height == height:
            return []
        self._free()
        s.frame_width = (width + 15) & ~0xF
        s.frame_height = (height + 15) & ~0xF
        s.pic_width, s.pic_height = width, height
        self.ctx = self.lib.th_encode_alloc(ctypes.byref(s))
        if not self.ctx:
            raise RuntimeError("[theora] Failed to create encoding context")
        self._keyframe_frequency()
        comment = ThComment()
        self.lib.th_comment_init(ctypes.byref(comment))
        self._vendor = ctypes.c_char_p(VENDOR)       # kept alive; never freed by libtheora
        comment.vendor = self._vendor
        op = OggPacket()
        self.stream_header = []
        while self.lib.th_encode_flushheader(self.ctx, ctypes.byref(comment), ctypes.byref(op)) > 0:
            self.stream_header.append(self._packet(header, op))
        return list(self.stream_header)

    def encode(self, header, rgb: np.ndarray):
        h, w, _ = rgb.shape
        out = self._ensure(header, w, h)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        fw, fh = int(self.setup.frame_width), int(self.setup.frame_height)
        if (fw, fh) != (w, h):
            padded = np.zeros((fh, fw, 3), np.uint8)
            padded[:h, :w] = bgr
            bgr = padded
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y = np.ascontiguousarray(ycrcb[:, :, 0])
        cr = np.ascontiguousarray(cv2.pyrDown(np.ascontiguousarray(ycrcb[:, :, 1])))
        cb = np.ascontiguousarray(cv2.pyrDown(np.ascontiguousarray(ycrcb[:, :, 2])))
        planes = (ThImgPlane * 3)()
        for i, m in enumerate((y, cb, cr)):
            planes[i].width, planes[i].height = m.shape[1], m.shape[0]
            planes[i].stride = m.strides[0]
            planes[i].data = m.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        rval = self.lib.th_encode_ycbcr_in(self.ctx, planes)
        if rval:
            raise RuntimeError(f"[theora] {rval} in submitting uncompressed frame to encoder")
        op = OggPacket()
        while self.lib.th_encode_packetout(self.ctx, 0, ctypes.byref(op)) > 0:
            out.append(self._packet(header, op))
        return out

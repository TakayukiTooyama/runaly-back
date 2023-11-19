import numpy as np
import warnings
from dlclive.exceptions import DLCLiveWarning

try:
    import skimage

    SK_IM = True
except Exception:
    SK_IM = False

try:
    import cv2

    OPEN_CV = True
except Exception:
    from PIL import Image

    OPEN_CV = False
    warnings.warn(
        "OpenCV is not installed. Using pillow for image processing, which is slower.",
        DLCLiveWarning,
    )


def convert_to_ubyte(frame):
    if SK_IM:
        return skimage.img_as_ubyte(frame)
    else:
        return _img_as_ubyte_np(frame)


def resize_frame(frame, resize=None):
    if (resize is not None) and (resize != 1):
        if OPEN_CV:
            new_x = int(frame.shape[0] * resize)
            new_y = int(frame.shape[1] * resize)
            return cv2.resize(frame, (new_y, new_x))

        else:
            img = Image.fromarray(frame)
            img = img.resize((new_y, new_x))
            return np.asarray(img)

    else:
        return frame


def img_to_rgb(frame):
    if frame.ndim == 2:
        return gray_to_rgb(frame)

    elif frame.ndim == 3:
        return bgr_to_rgb(frame)

    else:
        warnings.warn(
            f"Image has {frame.ndim} dimensions. Must be 2 or 3 dimensions to convert to RGB",
            DLCLiveWarning,
        )
        return frame


def gray_to_rgb(frame):
    if OPEN_CV:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    else:
        img = Image.fromarray(frame)
        img = img.convert("RGB")
        return np.asarray(img)


def bgr_to_rgb(frame):
    if OPEN_CV:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    else:
        img = Image.fromarray(frame)
        img = img.convert("RGB")
        return np.asarray(img)


def _img_as_ubyte_np(frame):
    frame = np.array(frame)
    im_type = frame.dtype.type

    # check if already ubyte
    if np.issubdtype(im_type, np.uint8):
        return frame

    # if floating
    elif np.issubdtype(im_type, np.floating):
        if (np.min(frame) < -1) or (np.max(frame) > 1):
            raise ValueError("Images of type float must be between -1 and 1.")

        frame *= 255
        frame = np.rint(frame)
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    # if integer
    elif np.issubdtype(im_type, np.integer):
        im_type_info = np.iinfo(im_type)
        frame *= 255 / im_type_info.max
        frame[frame < 0] = 0
        return frame.astype(np.uint8)

    else:
        raise TypeError(
            "image of type {} could not be converted to ubyte".format(im_type)
        )


def decode_fourcc(cc):
    try:
        decoded = "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
    except:
        decoded = ""

    return decoded

import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, data_dict, feature):
        self.tlwh=np.asarray(tlwh, dtype=float)
        self.confidence=float(confidence)
        self.data_dict = data_dict
        self.feature=np.asarray(feature,dtype=np.float32)

    def get_data(self):
        return self.data_dict
    
    def get_confidence(self):
        return self.confidence
    
    def to_tlbr(self):
        #Conversion of bounding box details from (x,y,w,h) to (x_start,y_start,x_end,y_end)
        ret=self.tlwh.copy()
        ret[2:]+=ret[:2]
        return ret

    def to_xyah(self):
        #Conversion of bounding box details from (x,y,w,h) to (x_center,y_center,aspect ratio, height)
        #aspect ratio=width/height
        ret=self.tlwh.copy()
        ret[:2]+=ret[2:]/2
        ret[2]/=ret[3]
        return ret

import cv2
import numpy

def union_of_intersecting_components(mask_a, mask_b):
    intersection = cv2.bitwise_and(mask_a, mask_b)
    union = numpy.zeros_like(intersection)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(...)

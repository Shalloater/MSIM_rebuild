# import cv2
# import numpy as np
# import array_detect as ar
import array_illumination as ai
import matplotlib
# 设置后端为 Agg
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_path = "lake.tif"
    # image_path = "data/24-2-5frames.tif"
    ai.get_lattice_vectors(calibration_name=image_path)

    # _, image_all = cv2.imreadmulti(image_path, flags=cv2.IMREAD_UNCHANGED)
    # image = np.array(image_all)
    # ar.detect_dot_centers(image, weighted=True, verbose=False, show=True)







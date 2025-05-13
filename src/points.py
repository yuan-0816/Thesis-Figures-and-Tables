import numpy as np
import os

# ------------------------------- UWB Experiment ------------------------------ #
uwb_exp_ipt430m_homography_real_points = np.array(
    [
        [8.4, 3.65],    # LowerLeft
        [4.81, 0.69],   # LowerRight
        [5.21, 9.23],   # UpperLeft
        [1.65, 3.89],   # UpperRight
    ]
)

uwb_exp_ipt430m_homography_pixel_points = np.array(
    [
        [445, 126],     # LowerLeft
        [483, 369],     # LowerRight
        [64, 83],       # UpperLeft
        [50, 369],      # UpperRight
    ]
)

uwb_exp_ipt430m_pixel_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/uwb_exp/thermal_IPT430M_pixel_points.npy",
    )
)

uwb_exp_ipt430m_predict_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/uwb_exp/thermal_IPT430M_world_points.npy",
    )
)

uwb_exp_ipt430m_real_points = np.array(
    [
        [5.6, 10.7],
        [6.7, 9.28],
        [7.68, 8.0],
        [8.7, 6.9],
        [9.78, 6.0],
        [11.1, 4.8],
        [3.8, 7.5],
        [4.7, 6.6],
        [5.5, 5.7],
        [6.3, 4.8],
        [7.1, 4.1],
        [7.9, 3.1],
        [2.7, 5.78],
        [3.5, 5.0],
        [4.1, 4.2],
        [4.8, 3.6],
        [5.5, 2.8],
        [6.2, 2.0],
        [2.0, 4.3],
        [2.68, 3.7],
        [3.2, 3.1],
        [3.78, 2.6],
        [4.4, 2.0],
        [4.88, 1.25],
    ]
)

uwb_exp_ds4025ft_homography_real_points = np.array(
    [
        [118, 953],     # LowerLeft
        [137, 100],     # LowerRight
        [1242, 913],    # UpperLeft
        [1153, 157],    # UpperRight
    ]
)

uwb_exp_ds4025ft_homography_pixel_points = np.array(
    [
        [4.75, 6.86],   # LowerLeft
        [8.63, 8.11],   # LowerRight
        [5.46, 4.63],   # UpperLeft
        [9.05, 5.36],   # UpperRight
    ]
)

uwb_exp_ds4025ft_pixel_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/uwb_exp/thermal_DS4025FT_pixel_points.npy",
    )
)

uwb_exp_ds4025ft_predict_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/uwb_exp/thermal_DS4025FT_world_points.npy",
    )
)


uwb_exp_ds4025ft_real_points = np.array(
    [
        [8.8, 8.3],
        [9.0, 7.7],
        [9.2, 7.1],
        [9.4, 6.5],
        [9.6, 5.9],
        [9.8, 5.2],
        [7.18, 7.7],
        [7.2, 7.1],
        [7.4, 6.6],
        [7.5, 6.0],
        [7.6, 5.5],
        [7.8, 5.0],
        [5.7, 7.3],
        [5.8, 6.78],
        [5.9, 6.2],
        [6.1, 5.8],
        [6.2, 5.2],
        [6.4, 4.8],
        [4.6, 6.9],
        [4.7, 6.4],
        [4.8, 5.9],
        [4.9, 5.5],
        [5.0, 5.1],
        [5.1, 4.68],
    ]
)


# --------------------------- Distortion Experiment -------------------------- #
dis_exp_homography_real_points = np.array(
    [
        [5.164, 0.184], # LowerLeft
        [3.767, 2.802], # LowerRight
        [0.34, 0.203],  # UpperLeft
        [0.348, 3.015], # UpperRight
    ]
)

dis_exp_distortion_homography_pixel_points = np.array(
    [
        [115, 366],     # LowerLeft
        [485, 153],     # LowerRight
        [64, 73],       # UpperLeft
        [313, 50],      # UpperRight    
    ]
)

dis_exp_distortion_pixel_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_distortion_exp/thermal_IPT430M_Distortion_pixel_points.npy",
    )
)

dis_exp_distortion_predict_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_distortion_exp/thermal_IPT430M_Distortion_predict_world_points.npy",
    )
)

dis_exp_distortion_real_points = np.array(
    [
        [-1.049, 0.08],
        [-0.38, 1.311],
        [-0.035, 2.437],
        [0.618, 3.263],     # [0.0, 0.0], # 牆壁, 量不到
        [1.223, 4.321],     # [0.0, 0.0], # 牆壁, 量不到
        [2.688, 0.025],
        [2.924, 0.728],
        [3.166, 1.358],
        [3.364, 2.005],
        [3.589, 2.707],
        [4.142, 0.000],
        [4.283, 0.488],
        [4.447, 0.927],
        [4.588, 1.395],
        [4.796, 1.856],
        [4.933, -0.04],
        [5.043, 0.342],
        [5.15, 0.71],
        [5.289, 1.062],
        [5.441, 1.41],
    ]
)


dis_exp_undistortion_homography_pixel_points = np.array(
    [
        [112, 379],     # LowerLeft
        [505, 152],     # LowerRight
        [58, 67],       # UpperLeft
        [316, 49],      # UpperRight
    ]
)

dis_exp_undistortion_pixel_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_distortion_exp/thermal_IPT430M_UnDistortion_pixel_points.npy",
    )
)

dis_exp_undistortion_predict_points = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_distortion_exp/thermal_IPT430M_UnDistortion_predict_world_points.npy",
    )
)


dis_exp_undistortion_real_points = np.array(
    [
        [-0.577, 0.167],
        [-0.356, 1.273],
        [0.108, 2.353],
        [0.576, 3.428],  # [0.0, 0.0], # 牆壁, 量不到
        [0.917, 4.485],  # [0.0, 0.0], # 牆壁, 量不到
        [2.675, 0.074],
        [2.942, 0.718],
        [3.14, 1.344],
        [3.38, 1.97],
        [3.59, 2.60],
        [4.118, 0.038],
        [4.27, 0.5],
        [4.42, 0.927],
        [4.57, 1.39],
        [4.72, 1.83],
        [4.87, 0.018],
        [4.99, 0.36],
        [5.10, 0.72],
        [5.23, 1.08],
        [5.34, 1.40],
    ]
)


def show_points(points):
    for i in range(len(points)):
        print(f"points[{i}] = {points[i]}")





def test():
    # print("test uwb exp ipt430m")
    # show_points(uwb_exp_ipt430m_homography_real_points)
    # show_points(uwb_exp_ipt430m_homography_pixel_points)
    # show_points(uwb_exp_ipt430m_pixel_points)
    # show_points(uwb_exp_ipt430m_predict_points)
    # show_points(uwb_exp_ipt430m_real_points)

    # print("test uwb exp ds4025ft")
    # show_points(uwb_exp_ds4025ft_homography_real_points)
    # show_points(uwb_exp_ds4025ft_homography_pixel_points)
    # show_points(uwb_exp_ds4025ft_pixel_points)
    # show_points(uwb_exp_ds4025ft_predict_points)
    # show_points(uwb_exp_ds4025ft_real_points)

    print("test distortion exp")
    # show_points(dis_exp_homography_real_points)
    # show_points(dis_exp_distortion_homography_pixel_points)
    # show_points(dis_exp_distortion_pixel_points)
    # show_points(dis_exp_distortion_predict_points)
    # show_points(dis_exp_distortion_real_points)
    # show_points(dis_exp_undistortion_homography_pixel_points)
    # show_points(dis_exp_undistortion_pixel_points)
    show_points(dis_exp_undistortion_predict_points)
    show_points(dis_exp_undistortion_real_points)
    print("test done")


if __name__ == "__main__":
    test()
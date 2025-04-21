import numpy as np
import matplotlib.pyplot as plt
import os


# 載入資料
IPT430M_PIXEL_POINTS = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_IPT430M_pixel_points.npy",
    )
)

IPT430M_PREDICT_POINTS = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_IPT430M_world_points.npy",
    )
)


IPT430M_REAL_POINTS = np.array([
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
])

DS4025FT_PIXEL_POINTS = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_DS4025FT_pixel_points.npy",
    )
)

DS4025FT_PREDICT_POINTS = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_DS4025FT_world_points.npy",
    )
)

DS4025FT_REAL_POINTS = np.array([
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
    [5.1, 4.68]
])





def compare_points(real_points, predict_poits, real_points_name="real", predict_poits_name="predict", title="Compare Points"):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 畫布設置
    ax.set_title(title)
    ax.scatter(real_points[:, 0], real_points[:, 1], c="red", label=real_points_name)
    ax.scatter(predict_poits[:, 0], predict_poits[:, 1], c="blue", label=predict_poits_name)
    ax.set_xlabel("x (UWB)")
    ax.set_ylabel("y (UWB)")
    # ax.invert_yaxis()  # 影像 y 軸通常是反的
    ax.grid(False)
    ax.legend()
    ax.axis("equal")

    for i in range(len(real_points)):
        ax.annotate(str(i), (real_points[i, 0], real_points[i, 1]))
        ax.annotate(str(i), (predict_poits[i, 0], predict_poits[i, 1]))

    # plt.tight_layout()
    plt.show()





def circle_pattern():
    # 3D坐標點
    col = 3
    row = 11
    # 設定標定板尺寸（內部圓點數量）
    pattern_size = (col, row)  # 或者根據實際標定板調整

    obj_points = []  # 3D 世界座標
    img_points = []  # 2D 影像座標
    # 準備 3D 世界座標點
    objp = []
    for c in range(pattern_size[0]):
        for r in range(pattern_size[1]):
            if r % 2 == 0:
                objp.append([r, c * 2, 0])
            else:
                objp.append([r, c * 2 + 1, 0])

    objp = np.array(objp, np.float32)

    print(objp)

    index = np.lexsort((objp[:, 2], objp[:, 1], objp[:, 0]))

    # print(index)
    # 再按 y 排序
    # sorted_objp = sorted_by_x[np.argsort(sorted_by_x[:, 1])]

    # print(sorted_objp)
    ans = objp[index]
    print(ans)

    # # 畫出2D投影
    # plt.scatter(objp[:, 0], objp[:, 1], c='purple', marker='o')
    # plt.title("2D Projection of Points")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.grid(True)
    # plt.show()


def UWB_to_pixel(device="IPT430M"):

    # 載入資料
    pixel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_{device}_pixel_points.npy",
    )
    world_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/thermal_{device}_world_points.npy",
    )

    # 載入資料
    pixel_points = np.load(pixel_path)
    world_points = np.load(world_path)

    print("pixel_points", pixel_points)
    print("world_points", world_points)



    # 畫布設置
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ===== 畫像素座標圖 =====
    ax1.set_title("Pixel Coordinates")
    ax1.scatter(pixel_points[:, 0], pixel_points[:, 1], c="red", label="pixel")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.invert_yaxis()  # 影像 y 軸通常是反的
    ax1.grid(False)
    ax1.legend()
    ax1.axis("equal")

    # ===== 畫世界座標圖 =====
    ax2.set_title("World Coordinates")
    ax2.scatter(world_points[:, 0], world_points[:, 1], c="blue", label="world")
    ax2.set_xlabel("x (world)")
    ax2.set_ylabel("y (world)")
    ax2.grid(False)
    ax2.legend()
    ax2.axis("equal")

    for i in range(len(pixel_points)):
        ax1.annotate(str(i), (pixel_points[i, 0], pixel_points[i, 1]))
        ax2.annotate(str(i), (world_points[i, 0], world_points[i, 1]))

    plt.tight_layout()
    plt.show()



import cv2
import glob




class CalibrationData():
    def __init__(self, device, exp_num=1):
        self.device = device
        self.calibration_data = None
        self.calibration_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../data/calibration_data/{device}/exp{str(exp_num)}/calibration_data.npz",
        )
        self.calibration_data = np.load(self.calibration_data_path, allow_pickle=True)
        self.obj_points = self.calibration_data['obj_points']
        self.img_points = self.calibration_data['img_points']
        self.camera_matrix = self.calibration_data['camera_matrix']
        self.dist_coeffs = self.calibration_data['dist_coeffs']
        self.rvecs = self.calibration_data['rvecs']
        self.tvecs = self.calibration_data['tvecs']

        if 'ret' in self.calibration_data:
            self.ret = self.calibration_data['ret']
        else:
            # 如果沒有 ret，則設置為 None 或其他適當的值
            self.ret = None


    def show_calibration_data(self):
        print(f"fx: {self.camera_matrix[0][0]}")
        print(f"fy: {self.camera_matrix[1][1]}")
        print(f"cx: {self.camera_matrix[0][2]}")
        print(f"cy: {self.camera_matrix[1][2]}")
        print(f"k1: {self.dist_coeffs[0][0]}")
        print(f"k2: {self.dist_coeffs[0][1]}")
        print(f"p1: {self.dist_coeffs[0][2]}")
        print(f"p2: {self.dist_coeffs[0][3]}")
        print(f"k3: {self.dist_coeffs[0][4]}")

    def MeanReprojectionError(self):
        '''
        計算重投影誤差的平均值, opencv ret 的回傳值
        '''
        mean_error = 0
        for i in range(len(self.obj_points)):
            reprojected_points, _ = cv2.projectPoints(
                self.obj_points[i], 
                self.rvecs[i], 
                self.tvecs[i], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            error = cv2.norm(
                self.img_points[i], 
                reprojected_points, 
                cv2.NORM_L2
            )/len(reprojected_points)

            mean_error += error*error

        print(f"origin ret: {self.ret}")
        print( "total error: {}".format(np.sqrt(mean_error/len(self.obj_points)*len(self.obj_points[0]))))


        return mean_error
    

    def MeanReprojectionError2(self):
        '''
        計算重投影誤差的平均值, opencv 官網範例
        '''
        mean_error = 0
        # all_dist = 0
        for i in range(len(self.obj_points)):
            reprojected_points, _ = cv2.projectPoints(
                self.obj_points[i], 
                self.rvecs[i], 
                self.tvecs[i], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            error = cv2.norm(self.img_points[i], reprojected_points, cv2.NORM_L2) / len(self.obj_points[i])
            mean_error += error
            # all_dist += cv2.norm(self.img_points[i], reprojected_points, cv2.NORM_L2)

        # print(f"Total reprojection error: {all_dist/len(self.obj_points)/len(self.obj_points[0])}")
        print(f"Mean reprojection error: {mean_error/len(self.obj_points)}")



    # def MeanSquaredError(self):
    #     # 計算每張圖的 MSE
    #     all_mse = []
    #     for i in range(len(self.obj_points)):
    #         projected_points, _ = cv2.projectPoints(
    #             self.obj_points[i], 
    #             self.rvecs[i], 
    #             self.tvecs[i], 
    #             self.camera_matrix, 
    #             self.dist_coeffs
    #         )
    #         projected_points = projected_points.squeeze()
    #         img_pts = self.img_points[i].squeeze()

    #         mse = np.mean(np.sum((projected_points - img_pts)**2, axis=1))
    #         all_mse.append(mse)
    #         # print(f"Image {i+1} MSE: {mse:.4f}")

    #     overall_mse = np.mean(all_mse)
    #     # print(f"\n📌 Overall average MSE: {overall_mse:.4f}")
    #     print(f"\n📌 Overall average MSE: {overall_mse}")
    #     return overall_mse


    # def MaximumRelativeError(self):
    #     # 計算每張圖的 MRE
    #     all_mre = []
    #     for i in range(len(self.obj_points)):
    #         # 使用相機參數將 3D 世界座標投影為 2D 圖像點
    #         projected_points, _ = cv2.projectPoints(
    #             self.obj_points[i], 
    #             self.rvecs[i], 
    #             self.tvecs[i], 
    #             self.camera_matrix, 
    #             self.dist_coeffs
    #         )
    #         projected_points = projected_points.squeeze()
    #         img_pts = self.img_points[i].squeeze()

    #         # 計算每個點的相對誤差（相對誤差 = |預測點 - 實際點| / |實際點|）
    #         relative_errors = np.linalg.norm(projected_points - img_pts, axis=1) / np.linalg.norm(img_pts, axis=1)

    #         # 找出最大的相對誤差（MRE）
    #         mre = np.max(relative_errors)
    #         all_mre.append(mre)
    #         # print(f"Image {i+1} MRE: {mre:.4f}")

    #     # 計算所有圖片的平均 MRE
    #     overall_mre = np.mean(all_mre)
    #     max_mre = np.max(all_mre)
    #     print(f"\n📌 Overall average MRE: {overall_mre}")
    #     print(f"\n📌 Overall max MRE: {max_mre}")







def main():
    # compare_points(
    #     IPT430M_REAL_POINTS,
    #     IPT430M_PREDICT_POINTS,
    #     real_points_name="real",
    #     predict_poits_name="predict",
    #     title="IPT430M Compare Points",
    # )


    coin1 = CalibrationData("ds4025ft", exp_num=1)
    coin1.MeanReprojectionError2()
    # coin1.show_calibration_data()
    # coin2 = CalibrationData("ipt430m", exp_num=2)
    # coin2.MeanReprojectionError2()
    # coin3 = CalibrationData("ipt430m", exp_num=3)
    # coin3.MeanReprojectionError2()




if __name__ == "__main__":
    main()

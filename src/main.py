import numpy as np
import matplotlib.pyplot as plt
import os


result_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"../results",
)


def compare_points(
    real_points,
    predict_points,
    real_points_name="real",
    predict_points_name="predict",
    title="Compare Points",
):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 畫布設置
    ax.set_title(title)
    scatter_real = ax.scatter(
        real_points[:, 0], real_points[:, 1], c="red", label=real_points_name
    )
    scatter_predict = ax.scatter(
        predict_points[:, 0], predict_points[:, 1], c="blue", label=predict_points_name
    )
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    # ax.invert_yaxis()  # 影像 y 軸通常是反的
    ax.grid(True)
    ax.legend()
    ax.axis("equal")

    # for i in range(len(real_points)):
    #     ax.annotate(str(i), (real_points[i, 0], real_points[i, 1]))
    #     ax.annotate(str(i), (predict_points[i, 0], predict_points[i, 1]))

    # plt.tight_layout()
    # plt.show()

    return fig, ax, scatter_real, scatter_predict


def combine_two_fig(compare_fig1, compare_fig2):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # 使用 compare_points 函數來獲得圖形和軸
    fig1, ax1_inner, scatter_real_1, scatter_predict_1 = (
        compare_fig1  # 返回第一組點的圖形和軸
    )
    fig2, ax2_inner, scatter_real_2, scatter_predict_2 = (
        compare_fig2  # 返回第二組點的圖形和軸
    )

    # 直接將第一個子圖的數據繪製到 ax1
    ax1.scatter(
        scatter_real_1.get_offsets()[:, 0],
        scatter_real_1.get_offsets()[:, 1],
        c="red",
        label=scatter_real_1.get_label(),
        alpha=0.5
    )
    ax1.scatter(
        scatter_predict_1.get_offsets()[:, 0],
        scatter_predict_1.get_offsets()[:, 1],
        c="blue",
        label=scatter_predict_1.get_label(),
        alpha=0.5
    )
    ax1.set_title(fig1.axes[0].get_title())  # 使用標題
    ax1.set_xlabel(fig1.axes[0].get_xlabel())  # 使用x軸標籤
    ax1.set_ylabel(fig1.axes[0].get_ylabel())  # 使用y軸標籤
    ax1.legend()

    # 直接將第二個子圖的數據繪製到 ax2
    ax2.scatter(
        scatter_real_2.get_offsets()[:, 0],
        scatter_real_2.get_offsets()[:, 1],
        c="red",
        label=scatter_real_2.get_label(),
        alpha=0.5,
    )
    ax2.scatter(
        scatter_predict_2.get_offsets()[:, 0],
        scatter_predict_2.get_offsets()[:, 1],
        c="blue",
        label=scatter_predict_2.get_label(),
        alpha=0.5,
    )
    ax2.set_title(fig2.axes[0].get_title())  # 使用標題
    ax2.set_xlabel(fig2.axes[0].get_xlabel())  # 使用x軸標籤
    ax2.set_ylabel(fig2.axes[0].get_ylabel())  # 使用y軸標籤
    ax2.legend()

    # 調整佈局
    plt.tight_layout()
    plt.show()


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


class CalibrationData:
    def __init__(self, device, exp_num=1):
        self.device = device
        self.calibration_data = None
        self.calibration_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../data/calibration_data/{device}/exp{str(exp_num)}/calibration_data.npz",
        )
        self.calibration_data = np.load(self.calibration_data_path, allow_pickle=True)
        self.obj_points = self.calibration_data["obj_points"]
        self.img_points = self.calibration_data["img_points"]
        self.camera_matrix = self.calibration_data["camera_matrix"]
        self.dist_coeffs = self.calibration_data["dist_coeffs"]
        self.rvecs = self.calibration_data["rvecs"]
        self.tvecs = self.calibration_data["tvecs"]

        if "ret" in self.calibration_data:
            self.ret = self.calibration_data["ret"]
        else:
            # 如果沒有 ret，則設置為 None 或其他適當的值
            self.ret = None

    def show_calibration_data(self):
        print(f"fx: {self.camera_matrix[0][0]:.3f}")
        print(f"fy: {self.camera_matrix[1][1]:.3f}")
        print(f"cx: {self.camera_matrix[0][2]:.3f}")
        print(f"cy: {self.camera_matrix[1][2]:.3f}")
        print(f"k1: {self.dist_coeffs[0][0]:.3f}")
        print(f"k2: {self.dist_coeffs[0][1]:.3f}")
        print(f"p1: {self.dist_coeffs[0][2]:.3f}")
        print(f"p2: {self.dist_coeffs[0][3]:.3f}")
        print(f"k3: {self.dist_coeffs[0][4]:.3f}")

    def MeanReprojectionError(self):
        """
        計算重投影誤差的平均值, opencv ret 的回傳值
        """
        mean_error = 0
        for i in range(len(self.obj_points)):
            reprojected_points, _ = cv2.projectPoints(
                self.obj_points[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs,
            )
            error = cv2.norm(self.img_points[i], reprojected_points, cv2.NORM_L2) / len(
                reprojected_points
            )

            mean_error += error * error

        print(f"origin ret: {self.ret}")
        print(
            "total error: {}".format(
                np.sqrt(mean_error / len(self.obj_points) * len(self.obj_points[0]))
            )
        )

        return mean_error

    def MeanReprojectionError2(self):
        """
        計算重投影誤差的平均值, opencv 官網範例
        """
        mean_error = 0
        # all_dist = 0
        for i in range(len(self.obj_points)):
            reprojected_points, _ = cv2.projectPoints(
                self.obj_points[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs,
            )
            error = cv2.norm(self.img_points[i], reprojected_points, cv2.NORM_L2) / len(
                self.obj_points[i]
            )
            mean_error += error
            # all_dist += cv2.norm(self.img_points[i], reprojected_points, cv2.NORM_L2)

        # print(f"Total reprojection error: {all_dist/len(self.obj_points)/len(self.obj_points[0])}")
        mre = mean_error / len(self.obj_points)
        print(f"Mean reprojection error: {mre:.3f}")

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
    from points import (
        dis_exp_distortion_predict_points,
        dis_exp_distortion_real_points,
        dis_exp_undistortion_predict_points,
        dis_exp_undistortion_real_points,
    )

    combine_two_fig(
        compare_points(
            dis_exp_distortion_real_points,
            dis_exp_distortion_predict_points,
            real_points_name="real",
            predict_points_name="predict",
            title="Distortion Predict Points",
        ),
        compare_points(
            dis_exp_undistortion_real_points,
            dis_exp_undistortion_predict_points,
            real_points_name="real",
            predict_points_name="predict",
            title="Undistortion Predict Points",
        ),
    )

    # coin1 = CalibrationData("ipt430m", exp_num=3)
    # coin1.show_calibration_data()
    # coin1.MeanReprojectionError2()
    # coin1.show_calibration_data()
    # coin2 = CalibrationData("ipt430m", exp_num=2)
    # coin2.MeanReprojectionError2()
    # coin3 = CalibrationData("ipt430m", exp_num=3)
    # coin3.MeanReprojectionError2()


if __name__ == "__main__":
    main()

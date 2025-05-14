import numpy as np
import matplotlib.pyplot as plt
import os


result_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"../result"
)


class DistortionExp:
    def __init__(
        self,
        dis_real_points,
        dis_predict_points,
        undis_real_points,
        undis_predict_points,
        save_path=None,
    ):
        self.dis_real_points = dis_real_points
        self.dis_predict_points = dis_predict_points
        self.undis_real_points = undis_real_points
        self.undis_predict_points = undis_predict_points
        self.save_path = save_path

    def plot_points_compare(
        self,
        points1,
        points2,
        points1_name=None,
        points2_name=None,
        title=None,
        fig_size=(7, 7),
        alpha=0.5,
        grid=False,
        x_label=None,
        y_label=None,
        x_range=None,   # 設置 x 軸範圍
        y_range=None,   # 設置 y 軸範圍
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure  # 如果傳入了 ax，則獲取它的 figure

        if title is not None:
            ax.set_title(title)
            # fig.suptitle(title, fontsize=12)

        scatter_points1 = ax.scatter(
            points1[:, 0], points1[:, 1], c="red", label=points1_name, alpha=alpha
        )
        scatter_points2 = ax.scatter(
            points2[:, 0], points2[:, 1], c="blue", label=points2_name, alpha=alpha
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(grid)
        ax.axis("equal")

        ax.set_aspect('equal', adjustable='box')
        # 如果 x_range 和 y_range 被提供，設置範圍
        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_ylim(y_range)
        
        return fig, ax
    

    def plot_compare_distortion_and_undistortion(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        # fig.suptitle("Distortion vs Undistortion", fontsize=12)

        # distortion
        fig1, ax1 = self.plot_points_compare(
            points1=self.dis_real_points, 
            points2=self.dis_predict_points, 
            title="Distortion",
            x_label="x (meters)",
            y_label="y (meters)",
            points1_name="Real Points",
            points2_name="Predicted Points",
            x_range=(-1.5, 6),  # 設置 x 軸範圍
            y_range=(-1.5, 6),  # 設置 y 軸範圍
            ax=ax1,
        )  

        # undistortion
        fig2, ax2 = self.plot_points_compare(
            points1=self.undis_real_points, 
            points2=self.undis_predict_points, 
            title="Undistortion",
            x_label="x (meters)",
            y_label="y (meters)",
            points1_name="Real Points",
            points2_name="Predicted Points",
            x_range=(-1.5, 6),  # 設置 x 軸範圍
            y_range=(-1.5, 6),  # 設置 y 軸範圍
            ax=ax2,
        )
        return fig, (ax1, ax2)

    
    def show_fig(self, *figures, save_path=None):
        # 確保 save_path 路徑存在
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        for fig, ax in figures:
            # 顯示圖形
            plt.figure(fig)

        # 顯示所有圖表
        plt.show()


    def caculate_error(self):
        # 計算誤差
        # 我要分別計算distortion和undistortion 的 x,y的誤差
        dis_x_error = np.abs(self.dis_real_points[:, 0] - self.dis_predict_points[:, 0])
        dis_y_error = np.abs(self.dis_real_points[:, 1] - self.dis_predict_points[:, 1])
        dis_x_mean_error = np.mean(dis_x_error)
        dis_y_mean_error = np.mean(dis_y_error)
        dis_x_max_error = np.max(dis_x_error)
        dis_y_max_error = np.max(dis_y_error)
        # 計算每個點的誤差
        dis_error = np.linalg.norm(self.dis_real_points - self.dis_predict_points, axis=1)
        dis_mean_error = np.mean(dis_error)
        dis_max_error = np.max(dis_error)

        print(f"--------------------Distortion--------------------")
        print(f"Distortion X Mean Error: {dis_x_mean_error:.4f}")
        print(f"Distortion Y Mean Error: {dis_y_mean_error:.4f}")
        print(f"Distortion X Max Error: {dis_x_max_error:.4f}")
        print(f"Distortion Y Max Error: {dis_y_max_error:.4f}")
        print(f"Distortion Mean Error: {dis_mean_error:.4f}")
        print(f"Distortion Max Error: {dis_max_error:.4f}")



        undis_x_error = np.abs(self.undis_real_points[:, 0] - self.undis_predict_points[:, 0])
        undis_y_error = np.abs(self.undis_real_points[:, 1] - self.undis_predict_points[:, 1])
        undis_x_mean_error = np.mean(undis_x_error)
        undis_y_mean_error = np.mean(undis_y_error)
        undis_x_max_error = np.max(undis_x_error)
        undis_y_max_error = np.max(undis_y_error)
        undis_error = np.linalg.norm(self.undis_real_points - self.undis_predict_points, axis=1)
        undis_mean_error = np.mean(undis_error)
        undis_max_error = np.max(undis_error)
        print(f"--------------------Undistortion--------------------")
        print(f"Undistortion X Mean Error: {undis_x_mean_error:.4f}")
        print(f"Undistortion Y Mean Error: {undis_y_mean_error:.4f}")
        print(f"Undistortion X Max Error: {undis_x_max_error:.4f}")
        print(f"Undistortion Y Max Error: {undis_y_max_error:.4f}")
        print(f"Undistortion Mean Error: {undis_mean_error:.4f}")
        print(f"Undistortion Max Error: {undis_max_error:.4f}")

        return dis_mean_error, undis_mean_error, self.ret
        


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


    distortion_exp = DistortionExp(
        dis_predict_points=dis_exp_distortion_predict_points,
        dis_real_points=dis_exp_distortion_real_points,
        undis_predict_points=dis_exp_undistortion_predict_points,
        undis_real_points=dis_exp_undistortion_real_points,
    )

    distortion_exp.caculate_error()

    distortion_exp.show_fig(
        distortion_exp.plot_points_compare(
            points1=distortion_exp.dis_real_points, 
            points2=distortion_exp.dis_predict_points, 
            title="Distortion Points",
            x_label="x (meters)",
            y_label="y (meters)",
            points1_name="Real Points",
            points2_name="Predicted Points",
            x_range=(-1.5, 6),  # 設置 x 軸範圍
            y_range=(-1.5, 6),  # 設置 y 軸範圍
        ),
        # # undistortion
        distortion_exp.plot_points_compare(
            points1=distortion_exp.undis_real_points, 
            points2=distortion_exp.undis_predict_points, 
            title="Undistortion Points",
            x_label="x (meters)",
            y_label="y (meters)",
            points1_name="Real Points",
            points2_name="Predicted Points",
            x_range=(-1.5, 6),  # 設置 x 軸範圍
            y_range=(-1.5, 6),  # 設置 y 軸範圍
        ),
        distortion_exp.plot_compare_distortion_and_undistortion(),
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

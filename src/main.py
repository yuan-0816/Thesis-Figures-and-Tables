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


    def calculate_error(self):
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

        return dis_mean_error, undis_mean_error
        


class UwbExp:
    def __init__(
            self, 
            ipt_real_points=None, 
            ipt_predict_points=None,
            ds_real_points=None,
            ds_predict_points=None,
        ):
        self.ipt_real_points = ipt_real_points
        self.ipt_predict_points = ipt_predict_points
        self.ds_real_points = ds_real_points
        self.ds_predict_points = ds_predict_points

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
        fig.suptitle("IPT430M vs DS4025FT", fontsize=12)

        fig1, ax1 = self.plot_points_compare(
            points1=self.ipt_real_points,
            points2=self.ipt_predict_points,
            title="IPT430M",
            x_label="x (meters)",
            y_label="y (meters)",
            points1_name="Real Points",
            points2_name="Predicted Points",
            # x_range=(-1.5, 6),  # 設置 x 軸範圍
            # y_range=(-1.5, 6),  # 設置 y 軸範圍
            ax=ax1,
        )  

        fig2, ax2 = self.plot_points_compare(
            points1=self.ds_real_points,
            points2=self.ds_predict_points,
            title="DS4025FT",
            x_label="x (meters)",
            y_label="y (meters)",
            points1_name="Real Points",
            points2_name="Predicted Points",
            # x_range=(-1.5, 6),  # 設置 x 軸範圍
            # y_range=(-1.5, 6),  # 設置 y 軸範圍
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

    def calculate_error(self):
        # 計算誤差
        # 我要分別計算distortion和undistortion 的 x,y的誤差
        ipt_x_error = np.abs(self.ipt_real_points[:, 0] - self.ipt_predict_points[:, 0])
        ipt_y_error = np.abs(self.ipt_real_points[:, 1] - self.ipt_predict_points[:, 1])
        ipt_x_mean_error = np.mean(ipt_x_error)
        ipt_y_mean_error = np.mean(ipt_y_error)
        ipt_x_max_error = np.max(ipt_x_error)
        ipt_y_max_error = np.max(ipt_y_error)
        # 計算每個點的誤差
        ipt_error = np.linalg.norm(self.ipt_real_points - self.ipt_predict_points, axis=1)
        ipt_mean_error = np.mean(ipt_error)
        ipt_max_error = np.max(ipt_error)

        print(f"--------------------IPT430M--------------------")
        print(f"IPT430M X Mean Error: {ipt_x_mean_error:.4f}")
        print(f"IPT430M Y Mean Error: {ipt_y_mean_error:.4f}")
        print(f"IPT430M X Max Error: {ipt_x_max_error:.4f}")
        print(f"IPT430M Y Max Error: {ipt_y_max_error:.4f}")
        print(f"IPT430M Mean Error: {ipt_mean_error:.4f}")
        print(f"IPT430M Max Error: {ipt_max_error:.4f}")



        DS_x_error = np.abs(self.ds_real_points[:, 0] - self.ds_predict_points[:, 0])
        DS_y_error = np.abs(self.ds_real_points[:, 1] - self.ds_predict_points[:, 1])
        DS_x_mean_error = np.mean(DS_x_error)
        DS_y_mean_error = np.mean(DS_y_error)
        DS_x_max_error = np.max(DS_x_error)
        DS_y_max_error = np.max(DS_y_error)
        DS_error = np.linalg.norm(self.ds_real_points - self.ds_predict_points, axis=1)
        DS_mean_error = np.mean(DS_error)
        DS_max_error = np.max(DS_error)
        print(f"--------------------DS4025FT--------------------")
        print(f"DS4025FT X Mean Error: {DS_x_mean_error:.4f}")
        print(f"DS4025FT Y Mean Error: {DS_y_mean_error:.4f}")
        print(f"DS4025FT X Max Error: {DS_x_max_error:.4f}")
        print(f"DS4025FT Y Max Error: {DS_y_max_error:.4f}")
        print(f"DS4025FT Mean Error: {DS_mean_error:.4f}")
        print(f"DS4025FT Max Error: {DS_max_error:.4f}")








import cv2


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



import math

class UAV_Find_Fire:
    def __init__(self):
        omega = 2 * math.pi / 10  # 每10秒繞一圈的角速度
        k = 0.3 / (2 * math.pi)  # 螺旋擴展速率，每秒擴展0.3m
        max_radius = 1.0  # 當半徑到1.0時停止繪圖
        angle_max = 50  # 繪製的最大角度
        
        # 生成時間和角度
        time = np.linspace(0, angle_max, 1000)  # 時間範圍，從0到最大角度，總共1000個點
        angle = omega * time  # 計算角度
        spiral_radius = k * angle  # 計算螺旋半徑

        # 計算 x 和 y 的位置
        x = spiral_radius * np.sin(angle)
        y = spiral_radius * np.cos(angle)
        
        # 限制半徑最大值
        self.x = np.clip(x, -max_radius, max_radius)
        self.y = np.clip(y, -max_radius, max_radius)

    def plot_spiral(self):
        plt.figure(figsize=(6, 6))
        plt.plot(self.x, self.y, label="Spiral Path")
        plt.title("Spiral Path for Fire Source Search")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(False)
        plt.axis('equal')  # 保持X和Y軸的比例相同
        plt.legend()

        # 顯示圖表
        plt.show()


def main():

    # from points import (
    #     dis_exp_distortion_predict_points,
    #     dis_exp_distortion_real_points,
    #     dis_exp_undistortion_predict_points,
    #     dis_exp_undistortion_real_points,
    # )


    # distortion_exp = DistortionExp(
    #     dis_predict_points=dis_exp_distortion_predict_points,
    #     dis_real_points=dis_exp_distortion_real_points,
    #     undis_predict_points=dis_exp_undistortion_predict_points,
    #     undis_real_points=dis_exp_undistortion_real_points,
    # )

    # distortion_exp.calculate_error()

    # distortion_exp.show_fig(
    #     distortion_exp.plot_points_compare(
    #         points1=distortion_exp.dis_real_points, 
    #         points2=distortion_exp.dis_predict_points, 
    #         title="Distortion Points",
    #         x_label="x (meters)",
    #         y_label="y (meters)",
    #         points1_name="Real Points",
    #         points2_name="Predicted Points",
    #         x_range=(-1.5, 6),  # 設置 x 軸範圍
    #         y_range=(-1.5, 6),  # 設置 y 軸範圍
    #         fig_size=(9, 9),
    #     ),
    #     # # undistortion
    #     distortion_exp.plot_points_compare(
    #         points1=distortion_exp.undis_real_points, 
    #         points2=distortion_exp.undis_predict_points, 
    #         title="Undistortion Points",
    #         x_label="x (meters)",
    #         y_label="y (meters)",
    #         points1_name="Real Points",
    #         points2_name="Predicted Points",
    #         x_range=(-1.5, 6),  # 設置 x 軸範圍
    #         y_range=(-1.5, 6),  # 設置 y 軸範圍
    #         fig_size=(9, 9),
    #     ),
    #     distortion_exp.plot_compare_distortion_and_undistortion(),
    # )


    from points import (
        uwb_exp_ipt430m_homography_real_points,
        uwb_exp_ipt430m_homography_pixel_points,
        uwb_exp_ipt430m_pixel_points,
        uwb_exp_ipt430m_predict_points,
        uwb_exp_ipt430m_real_points,

        uwb_exp_ds4025ft_homography_real_points,
        uwb_exp_ds4025ft_homography_pixel_points,
        uwb_exp_ds4025ft_pixel_points,
        uwb_exp_ds4025ft_predict_points,
        uwb_exp_ds4025ft_real_points,
    )
    uwb_exp = UwbExp(
        ipt_real_points=uwb_exp_ipt430m_real_points,
        ipt_predict_points=uwb_exp_ipt430m_predict_points,
        ds_real_points=uwb_exp_ds4025ft_real_points,
        ds_predict_points=uwb_exp_ds4025ft_predict_points,
    )

    uwb_exp.calculate_error()

    uwb_exp.show_fig(
        uwb_exp.plot_compare_distortion_and_undistortion()
    )

    


    # coin1 = CalibrationData("ipt430m", exp_num=3)
    # coin1.show_calibration_data()
    # coin1.MeanReprojectionError2()
    # coin1.show_calibration_data()
    # coin2 = CalibrationData("ipt430m", exp_num=2)
    # coin2.MeanReprojectionError2()
    # coin3 = CalibrationData("ipt430m", exp_num=3)
    # coin3.MeanReprojectionError2()

    # UAV_Find_Fire().plot_spiral()



if __name__ == "__main__":
    main()

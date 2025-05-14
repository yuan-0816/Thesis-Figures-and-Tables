import os
import numpy as np
import matplotlib.pyplot as plt


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

        return dis_mean_error, undis_mean_error

def circle_pattern(col=3, row=11):
    '''
    生成圓形標定板順序
    :param col: 圓形標定板的列數
    :param row: 圓形標定板的行數
    :return: 生成的圓形標定板的座標點 np.array
    '''

    import numpy as np
    import matplotlib.pyplot as plt

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

    # print(objp)

    index = np.lexsort((objp[:, 2], objp[:, 1], objp[:, 0]))

    # print(index)
    # 再按 y 排序
    # sorted_objp = sorted_by_x[np.argsort(sorted_by_x[:, 1])]

    # print(sorted_objp)
    ans = objp[index]
    # # 畫出2D投影
    # plt.scatter(objp[:, 0], objp[:, 1], c='purple', marker='o')
    # plt.title("2D Projection of Points")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.grid(True)
    # plt.show()
    # print(ans)
    return ans



if __name__ == "__main__":
    circle_pattern()


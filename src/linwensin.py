import numpy as np
# import pandas as pd



# === 建立旋轉矩陣：Z軸對齊 + 自動翻轉修正 ===
def normal_to_rotation(normal, reference_z=np.array([0, 0, 1])):
    z_axis = normal / np.linalg.norm(normal)
    if np.dot(z_axis, reference_z) < 0:
        z_axis = -z_axis
    x_temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(x_temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


# === 轉換點位並計算角度誤差 ===
def transform_points_quat(faro_points, T, robot_quat_ref=None):
    positions = faro_points[:, :3]
    normals = faro_points[:, 3:6]
    homo = np.hstack((positions, np.ones((positions.shape[0], 1))))
    transformed_xyz = (T @ homo.T).T[:, :3]
    quat_list, angle_error_list = [], []

    for i, n in enumerate(normals):
        try:
            if np.isnan(n).any():
                raise ValueError("NaN in normal vector")
            rot_matrix = normal_to_rotation(n)
            r_faro = R.from_matrix(rot_matrix)
            q_faro = r_faro.as_quat()
            quat_list.append(q_faro)

            if robot_quat_ref is not None:
                q_robot = R.from_quat(robot_quat_ref[i])
                q_delta = q_robot * r_faro.inv()
                angle_error_list.append(q_delta.magnitude() * (180 / np.pi))
            else:
                angle_error_list.append(np.nan)

        except Exception as e:
            print(f"⚠️ 第 {i} 筆 normal 發生錯誤：{e}")
            quat_list.append([np.nan] * 4)
            angle_error_list.append(np.nan)

    return np.hstack(
        (
            transformed_xyz,
            np.array(quat_list),
            np.array(angle_error_list).reshape(-1, 1),
        )
    )


# === 四元數轉歐拉角 ===
def quaternion_to_euler_df(quat_df):
    r = R.from_quat(quat_df[["qx", "qy", "qz", "qw"]].values)
    euler_deg = r.as_euler("xyz", degrees=True)
    return pd.DataFrame(euler_deg, columns=["Rx", "Ry", "Rz"])


# === 建立剛體轉換矩陣（SVD）===
def estimate_transformation(faro_pts, robot_pts):
    centroid_faro = np.mean(faro_pts, axis=0)
    centroid_robot = np.mean(robot_pts, axis=0)
    Q = faro_pts - centroid_faro
    P = robot_pts - centroid_robot
    H = Q.T @ P
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t = centroid_robot - R_ @ centroid_faro
    T = np.eye(4)
    T[:3, :3] = R_
    T[:3, 3] = t
    return T

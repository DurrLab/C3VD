import numpy as np
import cv2

def generate_omnidirectional_ray_map(intrinsics):
    """
    Generates an array of normalized ray directions using the provided
    camera intrinsics and the camera model from 'A Toolbox for Easily
    Calibrating Omnidirectional Cameras' by Scaramuzza, et al. (2006).

    Args:
        intrinsics : dict
            contains intrinsics parameters width, height, cx, cy, e, f, g, a0, a2, a3, a4

    Returns:
        numpy array of normalized ray directions for each pixel (size [H x W x 3])
    """

    # pixel coordinates
    ix, iy = np.meshgrid(np.arange(intrinsics['width']), np.arange(intrinsics['height']))

    # screen space coordinates
    uvp_x = ix - intrinsics['cx']
    uvp_y = iy - intrinsics['cy']
    uvp = np.stack([uvp_x, uvp_y], axis=-1)

    # apply inverted stretch matrix
    stretchMat = np.array([[intrinsics["c"], intrinsics["d"]], [intrinsics["e"], 1.0]])
    inv_stretch_mat = np.linalg.inv(stretchMat)
    uvpp = np.einsum('ij,...j->...i', inv_stretch_mat, uvp)

    rho = np.sqrt(uvpp[..., 0]**2 + uvpp[..., 1]**2)

    # polynomial for z-component
    z = (intrinsics['a0'] +
        (intrinsics['a2'] * rho**2) +
        (intrinsics['a3'] * rho**3) +
        (intrinsics['a4'] * rho**4))

    # combine uvpp and z into ray vector
    rays = np.stack([uvpp[..., 0], uvpp[..., 1], z], axis=-1)

    # normalize rays to directions, avoid dividing by zero
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    ray_map = rays / norms

    return ray_map

def load_poses(file_path):
    """
    Loads homogenous camera-to-world transformation matrices from a pose file.

    Args:
        file_path : string
            path to the pose file.

    Returns:
        list of 4x4 transformation matrices as numpy arrays.
    """
    poses = []
    try:
        with open(file_path, 'r') as fp:
            for line in fp:
                pose = np.fromstring(line.strip(), dtype=float, sep=',')
                if pose.size == 16:
                    poses.append(pose.reshape(4, 4))
                else:
                    raise ValueError("Pose line does not contain 16 elements.")
    except Exception as e:
        print(f"Error loading poses: {e}")
    return poses

def load_depth_map(file_path):
    """
    Loads depth image from the C3VD dataset. Provided image should be 16-bit.

    Args:
        file_path : string
            file path to the depth map file.

    Returns:
        depth map where each pixel indicates distance along the camera z-axis in mm.
        Depth values equal to 0 mm or 100 mm are stored as nan.
    """

    depth = np.array(cv2.imread(str(file_path), -1))

    # set invalid or clipped depth values as NaN
    depth = np.where((depth == 0) | (depth == 2**16 - 1), np.nan, depth)

    # convert depth to float and scale it
    depth = (depth.astype(np.float32) / (2**16 - 1)) * 100

    return depth

def project_pc(ray_map, depth):
    """
    Reprojects the provided depth map to 3D points using the provided ray direction map.

    Args:
        ray_map : ndarray
            normalized ray directions for each pixel (size [HW x 3])
        depth : ndarray
            depth map in world units (size [H x W])

    Returns:
        array of reprojected 3D points (size [HW x 3])
    """

    # extract the ray direction components
    ray_x = ray_map[..., 0]
    ray_y = ray_map[..., 1]
    ray_z = ray_map[..., 2]

    # compute the 3D coordinates for each pixel
    x_3d = (depth * ray_x) / ray_z
    y_3d = (depth * ray_y) / ray_z
    z_3d = depth

    # stack the x, y, z coordinates into a 3D array
    points = np.stack([x_3d, y_3d, z_3d], axis=-1)

    # reshape to HW x 3
    points = points.reshape(-1, 3)

    return points

def transform_points(points_3d, transformation_matrix):
    """
    Transforms a point cloud using the provided homogeneous transformation.

    Args:
        points_3d : ndarray
            array of 3D points (size [N x 3])
        transformation_matrix : ndarray
            homogeneous transformation (size [4 x 4])

    Returns:
        transformed 3D points (size [N x 3])
    """

    # convert points_3d from [N x 3] to [N x 4] by adding a column of ones (homogeneous coordinates)
    points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # apply the transformation matrix (matrix multiplication)
    transformed_points_homogeneous = points_homogeneous.dot(transformation_matrix)
    
    # convert back to 3D coordinates by dropping the last column
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points

def export_to_ply(points_3d, colors=None, filename="output_point_cloud.ply"):
    """
    Stores the provided 3D points as a PLY file, optionally including color.

    Args:
        points_3d : ndarray
            array of 3D points (size [N x 3])
        colors : ndarray (optional)
            array of RGB colors (size [N x 3]) corresponding to the points
        filename : string
            optional filename to store the PLY as (default = "output_point_cloud.ply")
    """

    num_points = points_3d.shape[0]

    # open the file for writing
    with open(filename, 'w') as ply_file:
        # write the header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        if colors is not None:
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        # write the point data (x, y, z, [r, g, b])
        if colors is not None:
            for point, color in zip(points_3d, colors):
                ply_file.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        else:
            for point in points_3d:
                ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Point cloud saved to {filename}")

def index_to_rgb_gradient(idx):
    """
    Generates an RGB color triplet cooresponding to the given index.
    The color map follow a red->green->blue gradient.

    Args:
        idx : float
            Index of the color value. Must be in the range [0 1]

    Returns:
        ndarray with RGB color (size [1 x 3]).
    """

    if idx <= 0.5:
        color = np.array([1 - 2*idx, 2*idx, 0])  # R->G
    else:
        color = np.array([0, 1 - 2*(idx - 0.5), 2*(idx - 0.5)])  # G->B
    return color * 255

if __name__ == "__main__":

    # file path to C3VD data directory
    c3vd_dir = "./cecum_t4_b/"

    # depth frames to reproject
    frames_to_include = range(0,425,50)

    # reduce number of points by this factor
    cull_factor = 10
    
    # camera intrinsics parameter definition
    intrinsics = {
        "width": 1350,
        "height": 1080,
        "cx": 678.544839263292,
        "cy": 542.975887548343,
        "a0": 769.243600037458,
        "a1": 0.0,
        "a2": -0.000812770624150226,
        "a3": 6.25674244578925e-07,
        "a4": -1.19662182144280e-09,
        "c": 0.999986882249990,
        "d": 0.00288273829525059,
        "e": -0.00296316513429569
    }

    # generate a ray map from the intrinsic parameters
    ray_map = generate_omnidirectional_ray_map(intrinsics)

    # load pose log
    poses = load_poses('{}pose.txt'.format(c3vd_dir))

    # iterate through each depth frame, reproject, and append to point cloud
    pointcloud = np.empty([0,3])
    pointcolors = np.empty([0,3])
    for n in range(0,len(frames_to_include)):

        # current frame index
        idx = frames_to_include[n]

        # load depth map
        depth = load_depth_map('{}{:04d}_depth.tiff'.format(c3vd_dir,idx))

        # reproject depth map into camera coordinate system
        pl = project_pc(ray_map,depth)

        # transform to world coordinate system
        pw = transform_points(pl,poses[idx])

        # remove invalid points
        pw_cleaned = pw[~np.isnan(pw).any(axis=1)]

        # cull points
        pw_cleaned = pw_cleaned[::cull_factor,:]

        # append points to point cloud
        pointcloud = np.append(pointcloud,pw_cleaned,axis=0)

        # color for current frame index
        color = index_to_rgb_gradient(n/(len(frames_to_include)-1))
        
        # append colors to point cloud color map
        pointcolors = np.append(pointcolors,np.tile(color,(pw_cleaned.shape[0],1)),axis=0)

    # save point cloud to PLY file
    export_to_ply(pointcloud,pointcolors,"combined_pc.ply")
from __future__ import division
import torch
import torch.nn.functional as F

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    '''
    pixel_coords =
    tensor([[[[  0.,   1.,   2.,  ..., 413., 414., 415.],
              [  0.,   1.,   2.,  ..., 413., 414., 415.],
              [  0.,   1.,   2.,  ..., 413., 414., 415.],
              ...,
              [  0.,   1.,   2.,  ..., 413., 414., 415.],
              [  0.,   1.,   2.,  ..., 413., 414., 415.],
              [  0.,   1.,   2.,  ..., 413., 414., 415.]],

             [[  0.,   0.,   0.,  ...,   0.,   0.,   0.],
              [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
              [  2.,   2.,   2.,  ...,   2.,   2.,   2.],
              ...,
              [125., 125., 125.,  ..., 125., 125., 125.],
              [126., 126., 126.,  ..., 126., 126., 126.],
              [127., 127., 127.,  ..., 127., 127., 127.]],

             [[  1.,   1.,   1.,  ...,   1.,   1.,   1.],
              [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
              [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
              ...,
              [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
              [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
              [  1.,   1.,   1.,  ...,   1.,   1.,   1.]]]])
    '''


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    计算图片网格中每个点对应的相机坐标
    像素坐标: 左上角为 (0, 0, 1), 右下角为 (W-1, H-1, 1)
    相机坐标: 由 p = (1/depth) * K @ PP 得到 PP = depth * K^{-1} @ p

    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """

    # 构造 current_pixel_coords, 使其每一列对应图片中每个点的像素坐标 (x,y,1)^T
    # current_pixel_coords =
    # tensor([[[  0.,   1.,   2.,  ..., 413., 414., 415.],
    #         [  0.,   0.,   0.,  ..., 127., 127., 127.],
    #         [  1.,   1.,   1.,  ...,   1.,   1.,   1.]],
    #
    #         ... # batch 中余下三组
    #       ])
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]

    # 然后将 current_pixel_coords 左乘 depth 和 K^{-1} 变成相机坐标
    # p = (1/depth) * K @ P => P = depth * K^{-1} @ p
    # TODO 怎么保证乘以 intrinsics_inv 得到的 cam_coords 恰好都落在 [-1,1] 之间, 已经 Z 坐标的处理
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)

    return cam_coords * depth.unsqueeze(1) # torch.Size([4, 3, 128, 416])


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        注意变换矩阵 proj_c2p_rot 和 proj_c2p_tr 已经被 K 左乘过了
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    # cam_coords 是图片网格中每个位置在世界坐标下的 (X,Y,Z) 位置
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3) # TODO 为什么要给 Z 一个最小值, 和 normalizaiton 有关吗

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2)


def euler2mat(angle):
    # TODO euler2mat: convert euler angles to rotation matrix
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    # the source image (where to sample pixels): 即基于 depth, pose, intrinsics 所算得的 (x,y), 在 source image 里采样.

    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    # target 图片网格中每个点所对应的相机坐标
    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    # target to source 的 6DoF 变换矩阵
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    # source 图片网格中每个点对应的像素坐标 - project
    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr, intrinsics)  # [B,H,W,2]

    # 使用计算出的像素坐标, 从 source image 中采样 - warp
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points

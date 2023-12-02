'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Lab Sheet 1
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

# print("here", flush=True)
if __name__ == '__main__': 
    bDisplayAxis = True

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    # 这个是创建一个支持平面，用来放置所有的球体
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    name_list.append('sphere_r')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -4],
                     [0, 1, 0, 2],
                     [0, 0, 1, -2],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_g')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -7],
                     [0, 1, 0, 2],
                     [0, 0, 1, 3],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_b')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, 4],
                     [0, 1, 0, 1.5],
                     [0, 0, 1, 4],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])


    #########################################
    '''
    Question 2: Add another sphere to the scene

    Write your code here to define another sphere
    in world coordinate frame
    '''
    # name_list.append('sphere_world')
    # sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=1.)
    # mesh_list.append(sph_mesh)
    # ## the tranlsation part doesn't need to be exactly the same
    # ## as long as the new sphere is on the plane and not touching
    # ## other spheres
    # H_list.append(np.array(
    #                 [[1, 0, 0, 4],
    #                  [0, 1, 0, 1.],
    #                  [0, 0, 1, -3],
    #                  [0, 0, 0, 1]]
    #         ))
    # RGB_list.append([0., 0.5, 0.5])
    #########################################


    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if bDisplayAxis:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * 45*5/180.
    # theta = 0.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * 80/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * 45*5/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]


    ###################################################
    '''
    Extra Question: Add an extra camera view here

    Write your code here to define camera poses
    '''
    ## the students can have different theta values for H2_0
    ## but the theta for H2_1 should be the same or at least similar
    ## to keep the new camera roughly the same height as cam_0 and cam_1 
    theta = np.pi * -80/180.
    H2_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * 45*5/180.
    H2_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H2_wc = np.matmul(H2_1, H2_0)
    render_list.append((H2_wc, 'view2.png', 'depth2.png'))
    ###################################################


    # set camera intrinsics
    #设置相机内参：
    #K是一个相机的内参矩阵，包含了焦距和主点坐标，这些参数对于生成正确的图像是必要的。
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 415.69219381653056, 415.69219381653056, 319.5, 239.5)
    # print(K)
    # print(K.intrinsic_matrix.shape)
    print('Pose_0:\n', H0_wc)
    print('Pose_1:\n', H1_wc)
    print('Intrinsics\n:', K.intrinsic_matrix)
    # o3d.io.write_pinhole_camera_intrinsic("test.json", K)


    ############################################################
    '''
    Question 4 & 5: Add sphere w.r.t. camera coordinate frames

    Write your code here to define the sphere
    in the camera coordinate frame
    '''
    ## the students can have different x-values to place the sphere
    ## but the y-value and z-value should be the same
    # place the sphere at the centre of camera 0
    # 要求添加一个球体到场景中，使其在相机0拍摄的图像中心投影。在计算机视觉中，这意味着球体应该位于相机0的主轴(principal
    # axis)上。在相机坐标系中，主轴通常是指从相机光心(COP)
    # 出发，垂直于成像平面(image plane)的轴。
    # 要在图像中心投影一个三维物体，需要这个物体在相机坐标系中的Z轴上，而X和Y坐标是0。这样，当我们从相机的视角看时，物体会出现在视野的中心。
    # 我不知道上面这个英文注释为什么说x可以随便。我觉得应该是x和y都要是0，z是随便。
    vec = np.array([[0, 0, 20]])
    # # place the sphere to the right of the centre of camera 0
    # vec = np.array([[5, 0, 20]])
    # # place the above sphere on the plane
    # vec = np.array([[5, -2, 20]])
    # 可以不用纠结上面这个中心不中心的问题
    # 下面这个我觉得很关键，意思就是把相机坐标系下的点转换到世界坐标系下
    # 拿ho矩阵的逆矩阵乘以相机坐标系下的点，就可以得到世界坐标系下的点
    vec_w = transform_points(vec, np.linalg.inv(H0_wc))

    name_list.append('sphere_cam')
    RGB_list.append([0., 0.5, 0.5])
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    H_new = np.array(
                    [[1, 0, 0, vec_w[0, 0]],
                     [0, 1, 0, vec_w[0, 1]],
                     [0, 0, 1, vec_w[0, 2]],
                     [0, 0, 0, 1]]
            )
    sph_mesh.vertices = o3d.utility.Vector3dVector(
        transform_points(np.asarray(sph_mesh.vertices), H_new)
    )
    sph_mesh.paint_uniform_color([0., 0.5, 0.5])
    sph_mesh.compute_vertex_normals()
    obj_meshes.append(sph_mesh)
    ############################################################


    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()

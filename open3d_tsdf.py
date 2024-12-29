import open3d as o3d
import numpy as np

def visualize_reconstruction():
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open("20241214_204332.bag")
    
    metadata = bag_reader.metadata
    intrinsic = metadata.intrinsics
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = [0.1, 0.1, 0.1]
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,  # 5mm的体素大小
        sdf_trunc=0.01,      # 1cm的截断距离
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 相机位姿
    current_transformation = np.identity(4)
    previous_rgbd = None

    try:
        frame_count = 0
        while True:
            rgbd = bag_reader.next_frame()
            if rgbd is None:
                break
            
            color_np = np.asarray(rgbd.color)
            depth_np = np.asarray(rgbd.depth)
            
            color = o3d.geometry.Image(color_np)
            depth = o3d.geometry.Image(depth_np)
            
            current_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth,
                depth_scale=metadata.depth_scale,
                depth_trunc=1.5,
                convert_rgb_to_intensity=False
            )
            
            if frame_count > 0:
                option = o3d.pipelines.odometry.OdometryOption()
                option.depth_diff_max = 0.03
                option.depth_min = 0.3
                option.depth_max = 3.0
                
                success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    current_rgbd, previous_rgbd, intrinsic,
                    np.identity(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option
                )
                
                if success:
                    current_transformation = np.dot(current_transformation, trans)
                    volume.integrate(current_rgbd, intrinsic, current_transformation)
            else:
                volume.integrate(current_rgbd, intrinsic, current_transformation)
            
            previous_rgbd = current_rgbd
            frame_count += 1
            
            if frame_count % 5 == 0:
                print(f"\rProcessed {frame_count} frames", end="")
                mesh = volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                
                vis.clear_geometries()
                vis.add_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()
                
    except Exception as e:
        print("\nError processing frames:", e)
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\nTotal processed frames: {frame_count}")
        
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        print("Performing post-processing...")
        
        # 网格优化
        print("Optimizing mesh...")
        mesh = mesh.filter_smooth_simple(number_of_iterations=5)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        print("Saving result...")
        o3d.io.write_triangle_mesh("reconstruction_improved.ply", mesh)
        
        print(f"\nFinal mesh statistics:")
        print(f"Vertices: {len(np.asarray(mesh.vertices))}")
        print(f"Triangles: {len(np.asarray(mesh.triangles))}")
        
        vis.destroy_window()
        bag_reader.close()

def visualize_result():
    mesh = o3d.io.read_triangle_mesh("reconstruction_improved.ply")
    mesh.compute_vertex_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    render_option = vis.get_render_option()
    render_option.background_color = [0.1, 0.1, 0.1]
    render_option.point_size = 1.0
    render_option.show_coordinate_frame = True
    
    vis.add_geometry(mesh)
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    print("Starting reconstruction...")
    visualize_reconstruction()
    print("\nReconstruction completed! Showing result...")
    visualize_result()
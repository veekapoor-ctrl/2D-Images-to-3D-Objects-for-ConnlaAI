import os
import numpy as np
import cv2
import torch
import open3d as o3d
import trimesh

from transformers import DPTForDepthEstimation, DPTImageProcessor


def load_depth_model(device="cpu"):
    model_name = "Intel/dpt-large"  # good quality monocular depth model
    processor = DPTImageProcessor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def estimate_depth(image_bgr, processor, model, device="cpu"):
    # Convert BGR (OpenCV) to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        predicted_depth = outputs.predicted_depth

    # Resize depth to original image size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize depth to [0, 1]
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    return depth_norm


def depth_to_point_cloud(depth, fx=1.0, fy=1.0, cx=None, cy=None, scale=1.0):
    """
    Convert depth map to point cloud in a simple pinhole camera model.
    For a single image, we fake intrinsics to get a reasonable flat-ish plane.
    """
    h, w = depth.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Create pixel grid
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth * scale  # scale depth to world units

    # Back-project to 3D
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    points = np.stack((X, -Y, -Z), axis=-1)  # flip Y/Z for a more natural orientation
    points = points.reshape(-1, 3)
    return points


def create_mesh_from_depth(depth, color_image=None, voxel_size=0.002):
    # Convert depth to point cloud
    points = depth_to_point_cloud(depth, scale=1.0)

    # Remove invalid points
    mask = ~np.isnan(points).any(axis=1)
    points = points[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color_image is not None:
        # Match colors to points (simple flatten)
        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img_flat = img_rgb.reshape(-1, 3) / 255.0
        img_flat = img_flat[mask]
        pcd.colors = o3d.utility.Vector3dVector(img_flat)

    # Downsample for cleaner mesh
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # Remove low-density vertices (noise)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.02)
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

    # Optional: simplify mesh
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh


def open3d_to_trimesh(mesh_o3d):
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    # Colors (if present)
    if mesh_o3d.has_vertex_colors():
        colors = np.asarray(mesh_o3d.vertex_colors)
    else:
        colors = None

    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    return mesh_tm


def convert_jpeg_to_gltf(
    input_jpeg_path,
    output_gltf_path,
    device="cpu",
    voxel_size=0.002,
):
    if not os.path.exists(input_jpeg_path):
        raise FileNotFoundError(f"Input image not found: {input_jpeg_path}")

    # Load image
    image_bgr = cv2.imread(input_jpeg_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to load image. Check file path and format.")

    # Load depth model
    processor, model = load_depth_model(device=device)

    # Estimate depth
    depth = estimate_depth(image_bgr, processor, model, device=device)

    # Create mesh
    mesh_o3d = create_mesh_from_depth(depth, color_image=image_bgr, voxel_size=voxel_size)

    # Convert to trimesh
    mesh_tm = open3d_to_trimesh(mesh_o3d)

    # Export to glTF (embedded)
    mesh_tm.export(output_gltf_path, file_type="gltf")
    print(f"Saved glTF model to: {output_gltf_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JPEG image to 3D glTF model.")
    parser.add_argument("input", help="Path to input JPEG image")
    parser.add_argument("output", help="Path to output glTF file (e.g., model.gltf)")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for depth model",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.002,
        help="Voxel size for point cloud downsampling (smaller = more detail, heavier mesh)",
    )

    args = parser.parse_args()

    convert_jpeg_to_gltf(
        input_jpeg_path=args.input,
        output_gltf_path=args.output,
        device=args.device,
        voxel_size=args.voxel_size,
    )

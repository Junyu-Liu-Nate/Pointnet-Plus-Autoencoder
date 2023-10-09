import random
import numpy as np
import open3d as o3d

#%%
def sampleFromMesh(mesh_path, num_points_per_model=5000):
    mesh_data = load_mesh(mesh_path)

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']

    sampledPC = []

    total_area = sum(triangle_area([vertices[v_idx] for v_idx in face]) for face in faces)

    for face in faces:
        t1, t2, t3 = [vertices[i] for i in face]
        area = triangle_area([t1, t2, t3])
        num_points = max(1, int(num_points_per_model * (area / total_area)))

        for _ in range(num_points):
            p = sample_point_on_triangle(t1, t2, t3)
            sampledPC.append(p)
    
    scaledPC = scale_to_unit_sphere()
    sampledPC = np.array(sampledPC)

    return sampledPC

def sample_point_on_triangle(t1, t2, t3):
    r1 = random.random()
    r2 = random.random()

    p = (1 - np.sqrt(r1)) * t1 + np.sqrt(r1) * (1 - r2) * t2 + np.sqrt(r1) * r2 * t3
    return p

def triangle_area(triangle):
    t1, t2, t3 = triangle
    return 0.5 * np.linalg.norm(np.cross(t2 - t1, t3 - t1))

def is_inside_sphere(point, center, radius):
    distance = np.linalg.norm(point - center)
    return distance <= radius

def scale_to_unit_sphere(points):
    max_distance = max(np.linalg.norm(p) for p in points)
    scaled_points = [p / max_distance for p in points]
    return scaled_points

#%%
def fpsSample(points, num_sample):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform Farthest Point Sampling (Downsampling)
    downsampled_pcd = pcd.farthest_point_down_sample(num_sample)

    # Extract sampled points and normals
    sampled_points = np.asarray(downsampled_pcd.points)

    return sampled_points.tolist()

#%%
def load_mesh(mesh_path):
    """
        Input: Mesh file .off format
        Output: Dictionary for part {"vertices": [], "normals": []}
    """
    vertices = []
    faces = []

    with open(mesh_path, 'r') as file:
        # Read the header line
        header = file.readline().strip()
        if header != 'OFF':
            raise ValueError("Invalid OFF file format")

        # Read number of vertices, faces, and edges
        num_vertices, num_faces, _ = map(int, file.readline().split())
        # print(num_vertices, num_faces)

        # Read vertex coordinates
        for _ in range(num_vertices):
            vertex = list(map(float, file.readline().split()))
            vertices.append(np.array(vertex))

        # Read faces
        for _ in range(num_faces):
            face = list(map(int, file.readline().split()))
            num_vertices_per_face = face[0]
            if num_vertices_per_face != 3:
                raise ValueError("Only triangular faces are supported")
            faces.append(np.array(face[1:]))  # Store vertex indices

    return {
        'vertices': vertices,
        'faces': faces
    }
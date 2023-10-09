def sampleFromMesh(mesh_path, num_points_per_model=5000):
    mesh_data = load_mesh(mesh_path)

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']

    oriented_point_set = []

    total_area = sum(triangle_area([vertices[v_idx] for v_idx in face]) for face in faces)

    for face in faces:
        t1, t2, t3 = [vertices[i] for i in face]
        area = triangle_area([t1, t2, t3])
        num_points = int(num_points_per_model * (area / total_area))

        normal = np.cross(t2 - t1, t3 - t1)  # Calculate the face normal
        normal = normalize_vector(normal)

        for _ in range(num_points):
            p = sample_point_on_triangle(t1, t2, t3)
            oriented_point_set.append((p, normal))

    # Separate vertices and normals
    vertices_list = [p for p, _ in oriented_point_set]
    normals_list = [n for _, n in oriented_point_set]

    # Uniformly scale the point set to fit a sphere with diameter 1
    scaled_vertices_list = scale_to_unit_sphere(vertices_list)

    result_dict = {
        'vertices': scaled_vertices_list,
        'normals': normals_list
    }

    return result_dict

def sample_point_on_triangle(t1, t2, t3):
    r1 = random.random()
    r2 = random.random()

    p = (1 - np.sqrt(r1)) * t1 + np.sqrt(r1) * (1 - r2) * t2 + np.sqrt(r1) * r2 * t3
    return p
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
import pymeshlab


def apply_filter_convexhull(input_mesh_path, output_mesh_path,):

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load mesh
    ms.load_new_mesh(input_mesh_path)

    # apply convex hull filter to the current selected mesh (last loaded)
    ms.generate_convex_hull()
    # alternatively:
    # ms.apply_filter('generate_convex_hull')

    assert ms.mesh_number() == 2

    # save the current selected mesh
    ms.save_current_mesh(output_mesh_path)



if __name__ == "__main__":
    input_mesh_path  = "D:/Callisto/repo/meshprocessing/samples/mesh.ply"
    output_mesh_path  = "D:/Callisto/repo/meshprocessing/samples/mesh_convexhull.ply"
    apply_filter_convexhull(input_mesh_path, output_mesh_path)
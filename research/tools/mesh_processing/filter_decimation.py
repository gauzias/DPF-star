import pymeshlab


def apply_filter_decimation(input_mesh_path, output_mesh_path,):

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load mesh
    ms.load_new_mesh(input_mesh_path)

    # apply convex hull filter to the current selected mesh (last loaded)
    ms.meshing_decimation_quadric_edge_collapse(
        #targetfacenum=1000, 
        targetperc = 0.2,
        preservenormal=True, 
        preservetopology=True, 
        qualitythr=0.3)


    # save the current selected mesh
    ms.save_current_mesh(output_mesh_path)


if __name__ == "__main__":
    input_mesh_path  = "D:/Callisto/repo/meshprocessing/samples/mesh.ply"
    output_mesh_path  = "D:/Callisto/repo/meshprocessing/samples/mesh_decimation_02.ply"
    apply_filter_decimation(input_mesh_path, output_mesh_path)
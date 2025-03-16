import pymeshlab
import open3d as o3d
import numpy as np
import os

def snapshot(ply_file, output_image):
    # Charger le maillage avec PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_file)
    
    # Sauvegarder le maillage en tant que PLY après traitement
    processed_ply = os.path.join(os.path.dirname(output_image), "processed_mesh.ply")
    ms.save_current_mesh(processed_ply)

    print(f"Maillage transformé enregistré : {processed_ply}")

    # Charger le maillage avec Open3D pour l'affichage et la capture d'écran
    mesh = o3d.io.read_triangle_mesh(processed_ply)
    mesh.compute_vertex_normals()
    
    # Appliquer une rotation de 90° autour de l'axe Y et Z dans Open3D
    RY = mesh.get_rotation_matrix_from_axis_angle([0, -np.pi / 2, 0])  # 90° en radians autour de Y
    mesh.rotate(RY, center=(0, 0, 0))
    RZ = mesh.get_rotation_matrix_from_axis_angle([0, 0, -np.pi / 2])  # 90° en radians autour de Z
    mesh.rotate(RZ, center=(0, 0, 0))
    
    # Générer un maillage en wireframe (pour afficher les arêtes en noir)
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edges.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(edges.lines))])  # Couleur noire
    
    # Calculer la bounding box avec une marge
    bbox = mesh.get_axis_aligned_bounding_box()
    #min_bound = bbox.min_bound
    #max_bound = bbox.max_bound
    center = bbox.get_center()
    #extent = max_bound - min_bound
    #margin = 0.1 * np.max(extent)  # 10% de marge
    #new_min_bound = min_bound - margin
    #new_max_bound = max_bound + margin
    
    # Créer une visualisation Open3D avec ajustement du champ de vision
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)

    vis.add_geometry(mesh)
    vis.add_geometry(edges)  # Ajouter les arêtes noires
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(center)
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)  # Ajustement pour mieux cadrer le mesh
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capturer une image
    vis.capture_screen_image(output_image)
    vis.destroy_window()

    print(f"Image enregistrée : {output_image}")



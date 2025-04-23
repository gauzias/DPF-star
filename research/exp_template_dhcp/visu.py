import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import research.tools.rw as rw
import research.tools.snapshot.snap_mesh_v2 as snap_mesh

import numpy as np
# === Chemins de base ===
mesh_root = r"E:\dhcpSym_template"
texture_root = r"E:\dhcpSym_sulci_manual"
screenshot_root = r"E:\screen_dhcp"
os.makedirs(screenshot_root, exist_ok=True)

# === Dictionnaire des labels et des couleurs ===
label_names = {
    2: "S.C.", 3: "S.T.s.", 4: "S.Pe.C.inf.", 5: "S.Pe.C.sup.", 6: "S.Po.C.", 7: "F.I.P.",
    8: "S.F.sup.", 9: "S.F.inf.", 10: "S.T.i.", 11: "F.C.M.", 12: "F.Cal.", 13: "S.F.int.",
    14: "S.R.inf", 15: "F.Coll.", 16: "S.S.p.", 17: "S.Call.", 18: "F.P.O.",
    19: "S.Olf.", 20: "S.Or.", 21: "S.O.T.Lat.", 22: "S.C.LPC.",
    23: "S.F.int.", 24: "S.F.marginal", 25: "S.F.Orbitaire", 26: "S.Rh.",
    27: "F.I.P.sup.", 28: "S.Pa.sup."
}

value_color_dict = {
    0: "floralwhite", 1: "floralwhite", 2: "blue", 3: "green", 4: "orange",
    5: "purple", 6: "teal", 7: "gold", 8: "pink", 9: "cyan", 10: "magenta",
    11: "lime", 12: "brown", 13: "navy", 14: "salmon", 15: "chocolate",
    16: "olive", 17: "turquoise", 18: "indigo", 19: "darkred", 20: "darkblue",
    21: "darkgreen", 22: "coral", 23: "deepskyblue", 24: "darkorange",
    25: "orchid", 26: "lightseagreen", 27: "steelblue", 28: "slategray"
}

# === Boucle sur les semaines ===
for week in range(28, 45):
    week_str = f"week-{week}"
    mesh_filename = f"{week_str}_hemi-left_space-dhcpSym_dens-32k_wm.surf.gii"
    texture_filename = f"week-44_hemi-left_space-dhcpSym_dens-32k_wm.sulci_manual.shape.gii"  

    mesh_path = os.path.join(mesh_root,f"week-{week}", "hemi_left" , "dHCPSym", mesh_filename)
    texture_path = os.path.join(texture_root, texture_filename)

    if not os.path.exists(mesh_path):
        print(f"Mesh non trouvé : {mesh_path}")
        continue

    # Créer le dossier pour cette semaine
    output_folder = os.path.join(screenshot_root, week_str)
    os.makedirs(output_folder, exist_ok=True)

    # Charger mesh et texture
    mesh = rw.load_mesh(mesh_path)
    texture = rw.read_gii_file(texture_path)
    texture = np.round(texture).astype(int)
    print(np.unique(texture))
    # Screenshot
    snap_mesh.capture_colored_mesh_snapshots(
        input_mesh=mesh_path,
        scalars=texture,
        output_path=output_folder,
        colormap_type="custom",
        colormap=None,
        custom_dict=value_color_dict
    )

    # Générer légende
    legend_path = os.path.join(output_folder, "label_legend.png")
    patches = []
    for val, name in label_names.items():
        color = value_color_dict.get(val, "gray")
        label = f"{val}: {name}"
        patches.append(mpatches.Patch(color=color, label=label))

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.legend(handles=patches, loc='center left', frameon=False)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Screenshot + légende enregistrés pour {week_str}")

# Introduction
Here, we share all codes relativ to the paper : New scale-invariant sulcal depth measure : A response to the conceptual and methodological problems of sulcal depth estimation

# Organisation of the Repo
DPF-STAR/
├── app/
│   ├── compute_curvature.py
│   └── compute_dpfstar.py
│   └── visualiser.py
├── fonctions/
│   ├── curvature.py
│   ├── dpf.py
│   ├── dpfstar.py
│   ├── laplacian.py
│   ├── rw.py
│   ├── topology.py
│   └── texture.py
├── scripts/
│   ├── scripts_EXP1.py
│   ├── scripts_EXP2.py
│   ├── scripts_EXP3.py
│   ├── scripts_EXP4.py
└── setup.py


# Workspace configuration
1) Change the file .vscode/settings.json with the correct work directory
2) Change the file .vscode/launc.json with the correct work directory
3) Change the file config/py with the correct work directory

# How to use the app
## Configuration
1. Navigue dans le dossier racine de ton projet :
```bash
cd path/to/projet
```
2. upgrade pip et setuptools
```bash
pip install --upgrade setuptools pip
```
3. Installe le package en mode développement avec la commande :
```bash
pip install -e .
```
## Utilisation
the App allows you to compute curvature, the dpfstar depth of a mesh. The app allows you to to have a quick visualiser of the texture applied on the mesh.

### How to compute curvature
To compute curvature of a mesh, write the following command in your terminal : 
```python
python -m app.compute_curvature {path/to/your/mesh}
```

### How to compute DPF-star
To compute DPF-star of a mesh, write the following command in your terminal : 
```python
python -m app.compute_dpfstar {path/to/your/mesh} --curvature {curvature path}
```
if the argument curvature is not given, the code will consider the curvature in the corresponding folder or computing it.

### How to visualize 
```python
python -m app.visualizer {path/to/your/mesh} --texture {texture path}
```


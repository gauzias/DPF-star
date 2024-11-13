# DPF-STAR Repository

## Introduction
This repository contains the code associated with the paper: **"New Scale-Invariant Sulcal Depth Measure: A Response to the Conceptual and Methodological Problems of Sulcal Depth Estimation."** 

The code here provides tools to compute curvature and a novel DPF-STAR depth measure for brain surface meshes, addressing limitations in traditional sulcal depth estimation methods.

## Repository Organization
The repository is organized as follows:

```plaintext
DPF-STAR/
├── app/                       # Application for using the DPF-STAR method on custom data
│   ├── compute_curvature.py   # Script for computing mesh curvature
│   ├── compute_dpfstar.py     # Script for computing the DPF-STAR depth
│   └── visualiser.py          # Script for visualizing textures on meshes
├── fonctions/                 # Utility functions used by the app
│   ├── curvature.py
│   ├── dpf.py
│   ├── dpfstar.py
│   ├── laplacian.py
│   ├── rw.py
│   ├── topology.py
│   └── texture.py
├── scripts/                   # Scripts used in experiments for the paper
│   ├── scripts_EXP1.py
│   ├── scripts_EXP2.py
│   ├── scripts_EXP3.py
│   └── scripts_EXP4.py
└── setup.py                   # Setup file for installing the package
```

## Workspace Configuration
Before running the app, adjust the workspace configuration to ensure smooth operation:

1. Update `.vscode/settings.json` with the correct project root directory path.
2. Update `.vscode/launch.json` with the correct project root directory path.
3. Adjust the `config.py` file to point to the correct project root directory.

## How to Use the App

### Configuration Steps
1. **Navigate to the project root directory:**
   ```bash
   cd path/to/DPF-STAR
   ```

2. **Upgrade pip and setuptools:**
   ```bash
   pip install --upgrade setuptools pip
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Usage Overview
The app allows you to:
- Compute the **curvature** of a mesh.
- Calculate the **DPF-STAR depth** of a mesh.
- Quickly visualize the **texture** applied on the mesh.

### Commands

#### 1. Compute Curvature
To compute the curvature of a mesh, run:
```bash
python -m app.compute_curvature {path/to/your/mesh}
```
Replace `{path/to/your/mesh}` with the actual path to your mesh file.

#### 2. Compute DPF-STAR Depth
To calculate the DPF-STAR depth of a mesh, use:
```bash
python -m app.compute_dpfstar {path/to/your/mesh} --curvature {curvature/path}
```
- If the `--curvature` argument is not specified, the script will look for a curvature file in the corresponding folder or compute it if necessary.

#### 3. Visualize Textures
To visualize textures on a mesh, use the following command:
```bash
python -m app.visualiser {path/to/your/mesh} --texture {texture/path}
```
Replace `{path/to/your/mesh}` and `{texture/path}` with the paths to your mesh and texture files, respectively.

---


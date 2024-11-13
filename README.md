# Introduction
Here, we share all codes relativ to the paper : New scale-invariant sulcal depth measure : A response to the conceptual and methodological problems of sulcal depth estimation

# Workspace configuration
1) Change the file .vscode/settings.json with the correct work directory
2) Change the file .vscode/launc.json with the correct work directory
3) Change the file config/py with the correct work directory

# How to compute curvature
To compute curvature of a mesh, write the following command in your terminal : 
```python
python .\compute_curvature.py .\meshes\mesh.gii
```

# How to compute DPF-star
To compute DPF-star of a mesh, write the following command in your terminal : 
```python
python .\compute_dpfstar.py {mesh path} --curvature {curvature path}
```
if the argument curvature is not given, the code will consider the curvature in the corresponding folder or computing it.

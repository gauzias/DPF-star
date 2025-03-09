# DPF-STAR Repository

[![DOI](https://zenodo.org/badge/887228134.svg)](https://doi.org/10.5281/zenodo.14163623)


## Introduction
This repository contains the code associated with the paper: **"New Scale-Invariant Sulcal Depth Measure: A Response to the Conceptual and Methodological Problems of Sulcal Depth Estimation."** 

The code here provides tools to compute curvature and a novel DPF-STAR depth measure for brain surface meshes, addressing limitations in traditional sulcal depth estimation methods.

## Table of Contents  
1. [Installation](#installation) 
2. [Repository Organization](#organization) 
3. [Worspace Configuration](#configuration)
4. [How to use the App](#app)  
5. [Data and Scripts](#scripts)


<a name="installation"/>

## Installation

Follow these steps to set up and run this project on your machine.

### Prerequisites

- Make sure **Conda** is installed on your machine. You can download Miniconda or Anaconda here: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)  
- Ensure that `bash` is available (commonly pre-installed on Linux and macOS, or via Git Bash on Windows).

### Installation Steps

#### Option 1 : automatic installation of the conda env (recommended)
1. Fork and Clone this repository:
```bash
git clone https://github.com/your-username/DPF-star .git
cd DPF-star
```
2. Run the installation script:
```bash
./install.sh
```
This script will:
* Create a Conda environment using the environment.yml file.
* Activate the newly created environment.

If the script does not execute, make sure it is executable:
 ```bash
chmod +x install.sh
```
#### Option 2 : manual installation of the conda env
If you prefer not to use the script, here are the manual steps:
1. Create the Conda environment from the environment.yml file:
```bash
conda env create -f environment.yml
```
2. Activate the Conda environment:
```bash
conda activate your_project_name
```

### Option 3 : editable mode.

If you want to contribute to this repository, you can install all the dependencies and the application in editable mode using 'pip install -e .'. This command installs the package as a symbolic link, allowing you to test changes locally as you develop without needing to reinstall the package. For more details about editable installations, see [the pip documentation on editable installs](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs).

1. Navigate to the project root directory:
```bash
cd path/to/DPF-STAR
```
2. Upgrade pip and setuptools:
```bash
pip install --upgrade setuptools pip
```
3. Install the package in development mode:
```bash
pip install -e .
```
You can run pip install -e . after creating either a Python virtual environment or a Conda environment. While Conda environments are larger in size, they offer more robust dependency management and support for non-Python packages. For maximum flexibility, it is recommended to create and activate a Conda environment first, then use pip install -e . to install the application in editable mode, allowing you to develop and test changes while benefiting from Conda's comprehensive package management.

<a name="organization"/>

## Repository Organization
There are 3 main files in this repo : 
(1) app : functions you can run on command line and allow you to use the dpf-star method for your own studies.
(2) scripts : the scripts used for the different experience introduced in the article.
(3) functions : functions useful both for the app and the scripts.
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

<a name="configuration"/>

## Workspace Configuration
Before running the app, adjust the workspace configuration to ensure smooth operation:

1. Update `.vscode/settings.json` with the correct project root directory path.
2. Update `.vscode/launch.json` with the correct project root directory path.
3. Adjust the `config.py` file to point to the correct project root directory.

<a name="app"/>

## How to Use the App

### Usage Overview
The app allows you to:
- Compute the **curvature** of a mesh.
- Calculate the **DPF depth** of a mesh.
- Calculate the **DPF-STAR depth** of a mesh.
- Quickly visualize the **texture** applied on the mesh.

### Commands

#### 1. Compute Curvature
To compute the curvature of a mesh, run:
```bash
python -m app.compute_curvature {path/to/your/mesh} --display
```
Replace `{path/to/your/mesh}` with the actual path to your mesh file.
- If the `--display` argument is added, the script will automatically run the dash code for visualising the mesh with the curvature texture.

Here a screenshot of what you get with the example mesh : 
<img src="./images/display_curvature.png" alt="curvature display" width="500"/>

#### 2. Compute DPF Depth
To calculate the DPF depth of a mesh, use:
```bash
python -m app.compute_dpf {path/to/your/mesh} --curvature {curvature/path} --display
```
- If the `--curvature` argument is not specified, the script will look for a curvature file in the corresponding folder or compute it if necessary.
- If the `--display` argument is added, the script will automatically run the dash code for visualising the mesh with the dpf texture.

Here a screenshot of what you get with the example mesh : 
<img src="./images/display_dpf.png" alt="dpf display" width="500"/>

#### 3. Compute DPF-STAR Depth
To calculate the DPF-STAR depth of a mesh, use:
```bash
python -m app.compute_dpfstar {path/to/your/mesh} --curvature {curvature/path} --display
```
- If the `--curvature` argument is not specified, the script will look for a curvature file in the corresponding folder or compute it if necessary.
- If the `--display` argument is added, the script will automatically run the dash code for visualising the mesh with the dpf-star texture.

Here a screenshot of what you get with the example mesh : 
<img src="./images/display_dpfstar.png" alt="dpfstar display" width="500"/>

#### 4. Visualize Textures
To visualize textures on a mesh, use the following command:
```bash
python -m app.visualiser {path/to/your/mesh} --texture {texture/path} 
```
Replace `{path/to/your/mesh}` and `{texture/path}` with the paths to your mesh and texture files, respectively.

you will get :
```bash
Dash is running on http://XXX.X.X.X:XXXX/

 * Serving Flask app 'visualizer'
 * Debug mode: on
```
simply copy paste the url http://XXX.X.X.X:XXXX/ is your internet browser

<a name="scripts"/>

## Data and scripts of the paper

### 1. Experience 1.

#### a. Dataset manually labelised

The informations relative to the datasets are stored in the folder ./data/dataset_EXP1.csv
<img src="./images/screen_dataset_EXP1.png" alt="data EXP1" width="500"/>

### Évaluation de la Robustesse de la Méthode DPF-star à la Résolution du Maillage

## 1. Génération de Maillages de Différentes Résolutions

Pour évaluer la robustesse de notre méthode en fonction de la résolution du maillage, nous avons généré un ensemble de maillages échantillonnés à différents niveaux de résolution. Nous avons défini trois niveaux de résolution en fonction de l'aire moyenne de Voronoï du maillage :

- **Haute résolution** : Aire de Voronoï = (...)
- **Résolution moyenne** : Aire de Voronoï = (...)
- **Basse résolution** : Aire de Voronoï = (...)

Les maillages ont été décimés à l'aide du logiciel **MeshLab** et du filtre *Quadric Based Edge Collapse*, basé sur la méthode de **Garland et al., 1997**. Ce filtrage réduit le nombre de sommets tout en préservant autant que possible la géométrie originale du maillage. Le code est disponible sur notre **repository GitHub**.

**Figure 1 : Maillages à Différentes Résolutions**

*Légende : Maillages décimés avec MeshLab.*

---

## 2. Comparaison des Distributions de Profondeur entre Différentes Résolutions

Nous avons calculé la **DPF-star** pour chaque version du maillage, puis comparé les distributions de profondeur à l'aide de plusieurs tests statistiques. **La figure 2** présente la visualisation de la profondeur pour chaque maillage et **la figure 3** montre les histogrammes correspondants.

### **Observation des Différences Visuelles**

- Les valeurs de profondeur sont **globalement équivalentes** entre les résolutions.
- Les **isolignes apparaissent plus fines** sur le maillage haute résolution, ce qui est attendu puisque la topologie est plus précise.
- Plus le maillage est décimé, moins il capture les variations fines de courbure.

### **Comparaison Quantitative**

Nous avons réalisé plusieurs tests statistiques pour comparer les distributions de profondeur entre les différents niveaux de résolution.

| Comparaison  | Test du **Chi2** (p-value) | Test de **Kolmogorov-Smirnov** (p-value) | **Bhattacharyya** | **Divergence KL** |
|-------------|----------------|----------------------|---------------|----------------|
| Mesh 1 vs Mesh 0.5  | 0.0259 (p=1) | 0.2174 (p=0.6601) | -7.4031 | 1642.9 |
| Mesh 1 vs Mesh 0.2  | 0.1475 (p=1) | 0.3913 (p=0.083)  | -6.9313 | 3904.8 |
| Mesh 0.5 vs Mesh 0.2 | 0.1414 (p=1) | 0.2609 (p=0.4218) | -6.5882 | 1136.18 |

**Interprétation** : Les distributions ne présentent **pas de différences significatives** (p > 0.05 dans les tests KS et Chi2). La divergence KL montre une différence numérique entre les distributions, mais cela ne signifie pas qu'elles sont statistiquement distinctes.

**Figure 2 : Visualisation de la DPF-star selon la Résolution du Maillage**

*Ligne 1 : Maillages à différentes résolutions. Ligne 2 : DPF-star. Colonnes : Résolution 1, 0.5, 0.2.*

**Figure 3 : Histogrammes des Distributions de Profondeur**

---

## 3. Projection des Profondeurs sur un Maillage Sous-échantillonné

Pour analyser plus finement la robustesse de notre méthode, nous avons projeté la profondeur calculée à partir du **maillage haute résolution** sur les maillages de **plus basse résolution**. Ensuite, nous avons comparé ces projections avec les valeurs de profondeur directement calculées à partir du maillage basse résolution.

**Observation** :
- Les variations de profondeur projetées correspondent bien aux **variations topologiques** du maillage.
- Cela confirme que la projection barycentrique capture fidèlement la structure originale du maillage.

**Figure 4 : Projection des Profondeurs sur un Maillage Basse Résolution**

### **Quantification des Erreurs**
Nous avons calculé les **erreurs quadratiques moyennes (MSE), normalisées (NRMSE) et absolues (MAE)** entre les profondeurs projetées et celles du maillage basse résolution.

| Comparaison | **MSE** | **NRMSE** | **MAE** |
|-------------|--------|--------|--------|
| D(0.5) vs D(1)proj(0.5) | 0.039 | 0.033 | 0.131 |

**Interprétation** : L'erreur quadratique moyenne et l'erreur normalisée sont faibles, indiquant une bonne **conservation des structures morphologiques** lors de la projection.

---

## 4. Conclusion

Nos résultats montrent que la méthode **DPF-star** est **robuste à la résolution du maillage** :
- Les valeurs de profondeur restent **cohérentes** à travers les différentes résolutions.
- L'analyse statistique ne montre **aucune différence significative** entre les distributions de profondeur.
- La projection des profondeurs haute résolution sur un maillage sous-échantillonné **reste fiable et préserve les variations locales**.

Ces résultats suggèrent que DPF-star peut être appliqué à des **maillages de résolutions variées** sans perte majeure d'information, ce qui la rend adaptée à des analyses multi-résolutions en imagerie médicale.

---

## 5. Références
- **M. Garland & P. Heckbert.** *Surface Simplification Using Quadric Error Metrics.* In Proceedings of SIGGRAPH 97.


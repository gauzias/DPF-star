import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import nibabel as nib
import numpy as np
from app.functions import rw as sio
import plotly.colors as pc
import argparse



PREDEFINED_COLORS = [
    '#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FFA500',
    '#FFC0CB', '#800080', '#00FFFF', '#808080', '#000000'
]

# Fonction pour lire un fichier GIFTI (scalars.gii)
def read_gii_file(file_path):
    try:
        gifti_img = nib.load(file_path)
        scalars = gifti_img.darrays[0].data
        return scalars
    except Exception as e:
        print(f"Erreur lors du chargement de la texture : {e}")
        return None

# Fonction pour convertir des couleurs RGB en hexadécimal
def convert_rgb_to_hex_if_needed(colormap):
    hex_colormap = []
    for color in colormap:
        if color.startswith('rgb'):
            rgb_values = [int(c) for c in color[4:-1].split(',')]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
            hex_colormap.append(hex_color)
        else:
            hex_colormap.append(color)
    return hex_colormap

# Création d'une colormap avec des traits noirs
def create_colormap_with_black_stripes(base_colormap, num_intervals=10, black_line_width=0.01):
    temp_c = pc.get_colorscale(base_colormap)
    temp_c_2 = [ii[1] for ii in temp_c]
    old_colormap = convert_rgb_to_hex_if_needed(temp_c_2)
    custom_colormap = []
    base_intervals = np.linspace(0, 1, len(old_colormap))

    num_intervals = len(old_colormap)
    for i in range(len(old_colormap) - 1):
        custom_colormap.append([base_intervals[i], old_colormap[i]])
        if i % (len(old_colormap) // num_intervals) == 0:
            black_start = base_intervals[i]
            black_end = min(black_start + black_line_width, 1)
            custom_colormap.append([black_start, 'rgb(0, 0, 0)'])
            custom_colormap.append([black_end, old_colormap[i]])
    custom_colormap.append([1, old_colormap[-1]])
    return custom_colormap

# Fonction pour créer la visualisation du maillage avec une colorbar
def plot_mesh_with_colorbar(vertices, faces, scalars=None, color_min=None, color_max=None, camera=None, show_contours=False, colormap='jet', use_black_intervals=False, center_colormap_on_zero=False):
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=False, hoverinfo='text', showscale=False
    )

    if scalars is not None:
        color_min = color_min if color_min is not None else np.min(scalars)
        color_max = color_max if color_max is not None else np.max(scalars)

        if center_colormap_on_zero:
            max_abs_value = max(abs(color_min), abs(color_max))
            color_min, color_max = -max_abs_value, max_abs_value

        if use_black_intervals:
            colorscale = create_colormap_with_black_stripes(colormap)
        else:
            colorscale = colormap

        fig_data.update(
            intensity=scalars,
            intensitymode='vertex',
            cmin=color_min,
            cmax=color_max,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Scalars",
                tickformat=".2f",
                thickness=30,
                len=0.9
            ),
            hovertext=[f'Scalar value: {s:.2f}' for s in scalars]
        )

    fig = go.Figure(data=[go.Mesh3d(**fig_data)])
    if show_contours:
        fig.data[0].update(contour=dict(show=True, color='black', width=2))

    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=camera
    ),
    height=900,
    width=1000,
    margin=dict(l=10, r=10, b=10, t=10))

    return fig

# Fonction pour obtenir les noms des colormaps disponibles dans Plotly, en filtrant celles contenant '__' ou 'swatches'
def get_colorscale_names(colormap_type):
    if colormap_type == 'sequential':
        return [name for name in pc.sequential.__dict__.keys() if '__' not in name and 'swatches' not in name]
    elif colormap_type == 'diverging':
        return [name for name in pc.diverging.__dict__.keys() if '__' not in name and 'swatches' not in name]
    elif colormap_type == 'cyclical':
        return [name for name in pc.cyclical.__dict__.keys() if '__' not in name and 'swatches' not in name]
    return []

# Créer des ticks clairs pour le slider
def create_slider_marks(color_min_default, color_max_default):
    return {str(i): f'{i:.2f}' for i in np.linspace(color_min_default, color_max_default, 10)}

# fonction pour discretiser la colormap
def discretize_colormap(colormap, num_colors=10):
    colors = pc.get_colorscale(colormap)
    discrete_colors = []
    for i in range(num_colors):
        fraction = i / (num_colors - 1)
        color = pc.sample_colorscale(colors, fraction)[0]  # Extraire la couleur de la liste imbriquée
        discrete_colors.append([fraction, color])  # Ajouter le fractionnement et la couleur
    return discrete_colors

# Fonction principale pour exécuter l'application
def run_dash_app(mesh_path, texture_path=None):
    # Charger le mesh
    mesh = sio.load_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces

    # Charger la texture (si fournie)
    scalars = read_gii_file(texture_path) if texture_path else None

    # Définir l'intervalle min et max par défaut des scalaires si disponibles
    color_min_default, color_max_default = (np.min(scalars), np.max(scalars)) if scalars is not None else (0, 1)

    # Créer l'application Dash
    app = dash.Dash(__name__)

    # Layout de l'application

    app.layout = html.Div([
        html.H1("Visualisation de maillage 3D avec color bar interactive", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Sélectionner le type de colormap"),
                dcc.Dropdown(
                    id='colormap-type-dropdown',
                    options=[
                        {'label': 'Sequential', 'value': 'sequential'},
                        {'label': 'Diverging', 'value': 'diverging'},
                        {'label': 'Cyclical', 'value': 'cyclical'}
                    ],
                    value='sequential',
                    clearable=False
                ),
                html.Label("Sélectionner une colormap"),
                dcc.Dropdown(
                    id='colormap-dropdown',
                    options=[{'label': cmap, 'value': cmap} for cmap in get_colorscale_names('sequential')],
                    value='Viridis',
                    clearable=False
                ),
                html.Label("Discrétiser la colormap"),
                dcc.Checklist(
                    id='toggle-discrete-colormap',
                    options=[{'label': 'Oui', 'value': 'on'}],
                    value=[],
                    inline=True
                ),
                html.Label("Afficher les isolignes"),
                dcc.Checklist(
                    id='toggle-contours',
                    options=[{'label': 'Oui', 'value': 'on'}],
                    value=[],
                    inline=True
                ),
                html.Label("Activer traits noirs"),
                dcc.Checklist(
                    id='toggle-black-intervals',
                    options=[{'label': 'Oui', 'value': 'on'}],
                    value=[],
                    inline=True
                ),
                html.Label("Centrer la colormap sur 0"),
                dcc.Checklist(
                    id='toggle-center-colormap',
                    options=[{'label': 'Oui', 'value': 'on'}],
                    value=[],
                    inline=True
                ),
                dcc.RangeSlider(
                    id='range-slider',
                    min=color_min_default,
                    max=color_max_default,
                    step=0.01,
                    value=[color_min_default, color_max_default],
                    marks=create_slider_marks(color_min_default, color_max_default),
                    vertical=True,
                    verticalHeight=640,
                    tooltip={"placement": "right", "always_visible": True}
                ),
                html.Button("Personnaliser ma colormap", id='custom-colormap-button', n_clicks=0, style={'margin-top': '20px'})
            ], style={'height': '640px', 'display': 'inline-block', 'margin-right': '10px'}),
            html.Div([
                dcc.Graph(id='3d-mesh')
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),
            # Modale pour personnaliser la colormap
            html.Div(
                id="custom-colormap-modal",
                style={"display": "none", "position": "fixed", "zIndex": 1000, "top": "10%", "left": "20%", "width": "60%", "height": "80%", "backgroundColor": "white", "border": "1px solid #ccc", "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"},
                children=[
                    html.Div(
                        style={"padding": "20px", "overflow": "auto", "height": "calc(100% - 60px)"},
                        children=[
                            html.H2("Personnaliser la Colormap", style={"textAlign": "center"}),
                            html.Div(id="color-intervals-container", children=[]),
                            html.Button("Ajouter une couleur", id="add-interval-btn", style={"marginTop": "10px", "display": "block", "marginLeft": "auto", "marginRight": "auto"}),
                            html.Div(
                                children=[
                                    html.Label("Liste des couleurs disponibles :"),
                                    html.Div(
                                        id="color-options",
                                        children=[html.Div(style={"display": "inline-block", "width": "30px", "height": "30px", "backgroundColor": color, "margin": "5px", "border": "1px solid black"}, title=color) for color in PREDEFINED_COLORS],
                                        style={"textAlign": "center"}
                                    )
                                ],
                                style={"marginTop": "20px"}
                            ),
                        ]
                    ),
                    html.Div(
                        style={"padding": "10px", "textAlign": "right", "borderTop": "1px solid #ccc"},
                        children=[
                            html.Button("Annuler", id="cancel-customization-btn", style={"marginRight": "10px"}),
                            html.Button("Appliquer", id="apply-customization-btn", style={"backgroundColor": "#28a745", "color": "white"})
                        ]
                    )
                ]
            )
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'})
    ])

    # Callback pour mettre à jour la figure Plotly
    @app.callback(
        Output('3d-mesh', 'figure'),
        [Input('range-slider', 'value'),
        Input('toggle-contours', 'value'),
        Input('toggle-black-intervals', 'value'),
        Input('colormap-dropdown', 'value'),
        Input('toggle-center-colormap', 'value'),
        Input('toggle-discrete-colormap', 'value')],
        [State('3d-mesh', 'relayoutData')]
)
    def update_figure(value_range, toggle_contours, toggle_black_intervals, selected_colormap, center_colormap, toggle_discrete, relayout_data):
        min_value, max_value = value_range
        camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None
        show_contours = 'on' in toggle_contours
        use_black_intervals = 'on' in toggle_black_intervals
        center_on_zero = 'on' in center_colormap
        use_discrete_colormap = 'on' in toggle_discrete

        # Discrétiser la colormap si l'option est activée
        colormap = discretize_colormap(selected_colormap, num_colors=10) if use_discrete_colormap else selected_colormap

        # Créer la figure avec les paramètres mis à jour
        fig = plot_mesh_with_colorbar(
            vertices, faces, scalars, color_min=min_value, color_max=max_value,
            camera=camera, show_contours=show_contours, colormap=colormap,
            use_black_intervals=use_black_intervals, center_colormap_on_zero=center_on_zero
        )
        return fig

    # Callback pour mettre à jour la liste des colormaps en fonction du type choisi
    @app.callback(
        Output('colormap-dropdown', 'options'),
        [Input('colormap-type-dropdown', 'value')]
    )
    def update_colormap_options(selected_type):
        return [{'label': cmap, 'value': cmap} for cmap in get_colorscale_names(selected_type)]
    
    @app.callback(
        Output('custom-colormap-modal', 'style'),
        [Input('custom-colormap-button', 'n_clicks'),
        Input('cancel-customization-btn', 'n_clicks')],
        [State('custom-colormap-modal', 'style')]
    )
    def toggle_modal(open_clicks, cancel_clicks, current_style):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_style
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'custom-colormap-button':
            return {"display": "block", "position": "fixed", "zIndex": 1000, "top": "10%", "left": "20%", "width": "60%", "height": "80%", "backgroundColor": "white", "border": "1px solid #ccc", "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"}
        elif trigger_id == 'cancel-customization-btn':
            return {"display": "none"}
        return current_style
    


    # Lancer l'application
    app.run_server(debug=True)

# Fonction d'entrée pour gérer les arguments de la ligne de commande
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lancer l'application Dash pour visualiser un maillage 3D.")
    parser.add_argument('mesh_path', type=str, help="Chemin vers le fichier maillage .gii")
    parser.add_argument('--texture', type=str, help="Chemin vers le fichier texture .gii", default=None)
    args = parser.parse_args()

    # Lancer l'application avec les chemins fournis
    run_dash_app(args.mesh_path, args.texture)
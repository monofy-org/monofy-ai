import trimesh
import numpy as np
from PIL import Image
import os

from utils.file_utils import random_filename


def simplify_mesh(input_file, target_faces=1000, output_format=None):
    # Determine input file type
    _, input_extension = os.path.splitext(input_file)
    input_extension = input_extension.lower()[
        1:
    ]  # Remove the dot and convert to lowercase

    # Load the mesh
    mesh = trimesh.load(input_file)

    # Handle scene if it's a GLB or other scene-based format
    if isinstance(mesh, trimesh.Scene):
        # Assume the first geometry in the scene is the one we want
        mesh = next(iter(mesh.geometry.values()))

    # Convert vertex colors to texture if necessary
    if mesh.visual.kind == "vertex":
        texture = mesh.visual.to_texture()
        mesh.visual = texture

    # Simplify the mesh while preserving UV coordinates
    simplified_mesh = mesh.simplify_quadratic_decimation(
        target_faces=target_faces, preserve_uv=True
    )

    # Re-apply vertex colors if the simplified mesh has a texture
    if simplified_mesh.visual.kind == "texture":
        texture_image = Image.fromarray(simplified_mesh.visual.material.image)

        uv = simplified_mesh.visual.uv
        colors = np.array(
            [
                texture_image.getpixel(
                    (u * texture_image.width, v * texture_image.height)
                )
                for u, v in uv
            ]
        )

        simplified_mesh.visual = trimesh.visual.ColorVisuals(
            simplified_mesh, vertex_colors=colors
        )

    # Determine output format
    if output_format is None:
        output_format = input_extension  # Use the same format as input if not specified

    # Generate a random filename with the specified format
    output_file = random_filename(output_format)

    # Export the simplified mesh
    simplified_mesh.export(output_file)

    return output_file

from submodules.HyperTile.hyper_tile import split_attention


def hypertile(func, unet, vae, tile_size=256, aspect_ratio=1, *args, **kwargs):
    with split_attention(
        vae,
        tile_size=tile_size,
        aspect_ratio=aspect_ratio,
    ):
        with split_attention(
            unet,
            tile_size=tile_size,
            aspect_ratio=aspect_ratio,
        ):
            return func(*args, **kwargs)

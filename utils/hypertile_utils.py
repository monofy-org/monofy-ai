from hyper_tile.hyper_tile import split_attention


def hypertile(pipe, tile_size=128, aspect_ratio=1, *args, **kwargs):
    with split_attention(
        pipe.vae,
        tile_size=tile_size,
        aspect_ratio=aspect_ratio,
    ):
        with split_attention(
            pipe.unet,
            tile_size=tile_size * 2,
            aspect_ratio=aspect_ratio,
            swap_size=2,
            disable=False,
        ):
            return pipe(*args, **kwargs)

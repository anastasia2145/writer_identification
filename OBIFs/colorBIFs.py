import numpy as np

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return int(value[0:0 + lv // 3], 16), int(value[2:2 + lv // 3], 16), int(value[4:4 + lv // 3], 16)


colorBifsMap = {
    1: hex_to_rgb("#ff0000"),
    2: hex_to_rgb("#778077"),
    3: hex_to_rgb("#ffff00"),
    4: hex_to_rgb("#80ff00"),
    5: hex_to_rgb("#00ff00"),
    6: hex_to_rgb("#00ff80"),
    7: hex_to_rgb("#002222"),
    8: hex_to_rgb("#0080ff"),
    9: hex_to_rgb("#0000ff"),
    10: hex_to_rgb("#8000ff"),
    11: hex_to_rgb("#ff00ff"),
    12: hex_to_rgb("#999999"),
    13: hex_to_rgb("#4caf50"),
    14: hex_to_rgb("#ffeb3b"),
    15: hex_to_rgb("#f44336"),
    16: hex_to_rgb("#e91e63"),
    17: hex_to_rgb("#9c27b0"),
    18: hex_to_rgb("#3f51b5"),
    19: hex_to_rgb("#9980ff"),
    20: hex_to_rgb("#aaffaa"),
    21: hex_to_rgb("#50ffaa"),
    22: hex_to_rgb("#ff8aaa"),
    23: hex_to_rgb("#99ff99"),
    30: hex_to_rgb("#ffffff")
}


def bif_to_color(bif):
    return colorBifsMap.get(bif)


color_bif_vector = np.frompyfunc(bif_to_color, 1, 3)


def bifs_to_color_image(bifs):
    color_image = np.empty((bifs.shape[0], bifs.shape[1], 3), dtype=np.uint8)

    for i in range(0, bifs.shape[0]):
        for j in range(0, bifs.shape[1]):
            color_image[i, j] = color_bif_vector(bifs[i, j])

    return color_image
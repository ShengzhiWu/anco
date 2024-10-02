import numpy as np
from PIL import Image
from anco import ImageSystem, PointTarget, ImageTarget

path = "tests/maps/"

# 载入地图
img = Image.open(path + 'map.png')
img_data = np.array(img).astype(np.float32) / 255.
map = img_data[:, :, 3].T

# 载入食物地图
img = Image.open(path + 'food.png')
img_data = np.array(img).astype(np.float32) / 255.
food = img_data[:, :, 3].T

targets = [
    PointTarget(chemical_id=0, location=[120.0, 500.0], radius=10.),
    ImageTarget(chemical_id=1, image=food, continuous_value=True, randomly_hit=True, diffusion=0.1)
]
ac = ImageSystem(map.shape, map, targets, create_window=True)

while ac.gui.running:
    for substep in range(10):
        ac.evolve_one_step()
    ac.visualize()
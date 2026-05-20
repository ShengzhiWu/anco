import numpy as np
from PIL import Image

# To facilitate testing, we have made it compatible to load packages from both the Python environment's library folder and the project folder.
# If you installed the package using `pip install anco`, you can simply write a single line:
# `from anco import ImageSystem, PointTarget, ImageTarget`
try:
    # Attempting to load the library from the library directory of the Python environment.
    from anco import ImageSystem, PointTarget, ImageTarget
except ImportError:
    # Try loading the library from the project folder. This should work if you cloned the complete repository from GitHub.
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
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

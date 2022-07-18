import os
from tqdm import trange

scenes=['Scar','Coffee','Scarf','Car','Easyship']

for i in trange(len(scenes)):
    scene=scenes[i]
    os.system(f"python val.py --config ./configs/{scene}.txt --ft_path=ckpts/{scene}.tar")
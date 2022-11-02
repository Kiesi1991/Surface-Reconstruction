from utils import *
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
from pathlib import Path

path_results = os.path.join('results', 'trainNN10')
file_path = os.path.join(path_results, '**', f'errors.pt')
paths = glob.glob(file_path, recursive=True)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)

for idx, path in enumerate(paths):
    errors = torch.load(path)
    splits = np.array_split(errors, 100)
    parameters_path = os.path.join(Path(path).parent.absolute(), 'parameters.txt')
    with open(parameters_path) as f:
        lines = f.readlines()
    parameters = ','.join(lines).replace('\n', '').replace("<class 'models.", "").replace("'>", "").replace("layers", "L").replace("mid_channels", "C").replace("BlockNet = ", "")
    #errors = [mean(np.clip(el, 0, 0.02).tolist()) for el in splits]
    try:
        #errors = [mean(sorted(el.tolist())[:-5]) for el in splits]
        #errors = [mean(el.tolist()) for el in splits]
        x = np.linspace(0, len(errors) - 1, len(errors))
        child = Path(path).parent.absolute().parts[-1]
        if int(child) <= 9:
            plt.plot(x, errors, label=child + "," + parameters)
    except:
        print(f'not used {path}')
    plt.ylim(bottom=0, top=0.0004)
    plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('Error')
plt.legend()
plt.savefig(os.path.join(path_results))
plt.close()
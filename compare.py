from utils import *
import matplotlib.pyplot as plt

path_results = os.path.join('results', 'optimization', '1')
file_path = os.path.join(path_results, '**', f'surface.pt')
paths = glob.glob(file_path, recursive=True)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)

for idx, path in enumerate(paths):
    surface = torch.load(path)
    height_profile_x_pred, height_profile_y_pred = getHeightProfile(surface, divide_by_mean=False)
    x = np.linspace(0, len(height_profile_x_pred) - 1, len(height_profile_x_pred))
    plt.plot(x, height_profile_x_pred, label=f'prediction {idx}')
plt.xlabel('pixels')
plt.ylabel('height')
plt.legend()
plt.title('profile in x-direction')

plt.subplot(1, 2, 2)

for idx, path in enumerate(paths):
    surface = torch.load(path)
    height_profile_x_pred, height_profile_y_pred = getHeightProfile(surface, divide_by_mean=False)
    y = np.linspace(0, len(height_profile_y_pred) - 1, len(height_profile_y_pred))
    plt.plot(y, height_profile_y_pred, label=f'prediction {idx}')
plt.xlabel('pixels')
plt.ylabel('height')
plt.legend()
plt.title('profile in y-direction')

plt.savefig(os.path.join(path_results, f'height-profiles.png'))
plt.close()

print('TheEnd')
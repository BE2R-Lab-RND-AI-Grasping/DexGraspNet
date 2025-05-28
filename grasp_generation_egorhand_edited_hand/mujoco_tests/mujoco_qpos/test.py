import numpy as np

# Заменить 'имя_файла.npy' на путь к твоему файлу
data = np.load('qpos_dataset.npy', allow_pickle=True)

print(data)
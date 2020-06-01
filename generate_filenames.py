import os

asps = []
for root, dirs, files in os.walk(r'/mnt/edisk/superpoint_graph/Sema3d/features/train'):
    for file in files:
        if file.endswith('.h5'):
            asps.append(file)



train_names = asps[0:110]
val_names = asps[111:117]
print(train_names, val_names)
import os

with open('train.txt', 'w') as f:
    for name in os.listdir('images/train'):
        if name.endswith('.jpg'):
            base = name.replace('.jpg', '.png')
            f.write(f'images/train/{name} masks/train/{base}\n')


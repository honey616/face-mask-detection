import os
import random
import shutil

IMG_SRC = "dataset/images"
XML_SRC = "dataset/annotations"

IMG_DST = "dataset/images"
XML_DST = "dataset/annotations"

# create split folders
for split in ["train", "val", "test"]:
    os.makedirs(f"{IMG_DST}/{split}", exist_ok=True)
    os.makedirs(f"{XML_DST}/{split}", exist_ok=True)

images = [f for f in os.listdir(IMG_SRC) if f.endswith(".png") or f.endswith(".jpg")]
random.shuffle(images)

total = len(images)
train_imgs = images[:int(0.7 * total)]
val_imgs   = images[int(0.7 * total):int(0.85 * total)]
test_imgs  = images[int(0.85 * total):]

def move(files, img_dst, xml_dst):
    for f in files:
        shutil.move(f"{IMG_SRC}/{f}", f"{img_dst}/{f}")
        shutil.move(
            f"{XML_SRC}/{f.replace('.png','.xml')}",
            f"{xml_dst}/{f.replace('.png','.xml')}"
        )

move(train_imgs, f"{IMG_DST}/train", f"{XML_DST}/train")
move(val_imgs,   f"{IMG_DST}/val",   f"{XML_DST}/val")
move(test_imgs,  f"{IMG_DST}/test",  f"{XML_DST}/test")

print("Dataset successfully split into train / val / test")


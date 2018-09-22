from PIL import Image
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: resize.py /detaset_path/ /output_path/")
        exit(0)

    image_dir = sys.argv[1]
    output_dir = sys.argv[2]

    imgfiles = []
    f = open(image_dir + "image_list.csv", "r")
    for line in f:
        filename = line.rstrip() # 改行文字を削除
        imgfiles.append(filename)
    f.close()

    for i in range(len(imgfiles)):
        img = Image.open(image_dir + imgfiles[i]).resize((128, 128), Image.LANCZOS)
        img = img.convert("RGB")
        img.save(output_dir + imgfiles[i], quality=95)
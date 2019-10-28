from PIL import Image
import os
for filename in os.listdir('./pneumonia'):
    im = Image.open('./pneumonia/'+filename)
    imFlip = im.transpose(Image.FLIP_LEFT_RIGHT)
    crop = im.crop((100,100,500,500))
    sizeCrop = crop.resize((600,600), Image.ANTIALIAS)
    cropFlip = crop.transpose(Image.FLIP_LEFT_RIGHT)
    sizeCrop.save('cropped' + filename)
    imFlip.save('fliped'+filename)
    cropFlip.save('cropFlip'+filename)



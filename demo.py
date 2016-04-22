"""
demo.py - Neural Style Transfer demo.
"""

# system imports
import argparse

# library imports
import caffe
import cv2

# local imports
from style import StyleTransfer

# argparse
parser = argparse.ArgumentParser(description = "Transfer the style.", usage = "demo.py -s <style_image> -c <content_image>")
parser.add_argument("-s", "--style-img", type = str, required = True, help = "the style image")
parser.add_argument("-c", "--content-img", type = str, required = True, help = "the content image")

# use googlenet as defaults
transferer = StyleTransfer("googlenet")

def st_api(img_style, img_content, callback = None):
    """
        Style transfer API.
    """

    # style transfer arguments
    args = {"length": 512, "ratio": 2e4, "n_iter": 16, "callback": callback, "init": "content"}
    
    # start style transfer
    transferer.transfer_style(img_style, img_content, **args)
    img_out = transferer.get_generated()

    return img_out

if __name__ == "__main__":
    args = parser.parse_args()
    
    # perform style transfer
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    result = st_api(img_style, img_content)

    # save the image
    art = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("art.jpg", art * 255)

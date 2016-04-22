"""
style.py - An implementation of "A Neural Algorithm of Artistic Style"
by L. Gatys, A. Ecker, and M. Bethge. http://arxiv.org/abs/1508.06576.

authors: Frank Liu - frank@frankzliu.com
         Dylan Paiton - dpaiton@gmail.com
last modified: 10/06/2015

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Frank Liu (fzliu) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Frank Liu (fzliu) BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# system imports
import argparse
import os
import sys
import timeit

# library imports
import caffe
import numpy as np
from scipy.fftpack import ifftn
from scipy.linalg.blas import sgemm
from scipy.misc import imsave
from scipy.optimize import minimize
from skimage import img_as_ubyte
from skimage.transform import rescale

# numeric constants
INF = np.float32(np.inf)
STYLE_SCALE = 1.2

# weights for the individual models
# assume that corresponding layers' top blob matches its name
VGG19_WEIGHTS = {"content": {"conv4_2": 1},
                 "style": {"conv1_1": 0.2,
                           "conv2_1": 0.2,
                           "conv3_1": 0.2,
                           "conv4_1": 0.2,
                           "conv5_1": 0.2}}
VGG16_WEIGHTS = {"content": {"conv4_2": 1},
                 "style": {"conv1_1": 0.2,
                           "conv2_1": 0.2,
                           "conv3_1": 0.2,
                           "conv4_1": 0.2,
                           "conv5_1": 0.2}}
GOOGLENET_WEIGHTS = {"content": {"conv2/3x3": 2e-4,
                                 "inception_3a/output": 1 - 2e-4},
                     "style": {"conv1/7x7_s2": 0.2,
                               "conv2/3x3": 0.2,
                               "inception_3a/output": 0.2,
                               "inception_4a/output": 0.2,
                               "inception_5a/output": 0.2}}
CAFFENET_WEIGHTS = {"content": {"conv4": 1},
                    "style": {"conv1": 0.2,
                              "conv2": 0.2,
                              "conv3": 0.2,
                              "conv4": 0.2,
                              "conv5": 0.2}}

# argparse
parser = argparse.ArgumentParser(description = "Transfer the style of one image to another.",
                                 usage = "style.py -s <style_image> -c <content_image>")
parser.add_argument("-s", "--style-img", type = str, required = True, help = "input style (art) image")
parser.add_argument("-c", "--content-img", type = str, required = True, help = "input content image")
parser.add_argument("-g", "--gpu-id", default = 0, type = int, required = False, help = "GPU device number")
parser.add_argument("-m", "--model", default = "vgg16", type = str, required = False, help = "model to use")
parser.add_argument("-i", "--init", default = "content", type = str, required = False, help = "initialization strategy")
parser.add_argument("-r", "--ratio", default = "1e4", type = str, required = False, help = "style-to-content ratio")
parser.add_argument("-n", "--num-iters", default = 512, type = int, required = False, help = "L-BFGS iterations")
parser.add_argument("-l", "--length", default = 512, type = float, required = False, help = "maximum image length")
parser.add_argument("-v", "--verbose", action = "store_true", required = False, help = "print minimization outputs")
parser.add_argument("-o", "--output", default = None, required = False, help = "output path")


def _compute_style_grad(F, G, G_style, layer):

    """
        Compute style gradient and loss from activation features.
    """

    # compute loss and gradient
    (Fl, Gl) = (F[layer], G[layer])
    c = Fl.shape[0]**(-2) * Fl.shape[1]**(-2)
    El = Gl - G_style[layer]
    loss = c / 4 * (El**2).sum()
    grad = c * sgemm(1.0, El, Fl) * (Fl>0)

    return loss, grad

def _compute_content_grad(F, F_content, layer):

    """
        Compute content gradient and loss from activation features.
    """

    # compute loss and gradient
    Fl = F[layer]
    El = Fl - F_content[layer]
    loss = (El**2).sum() / 2
    grad = El * (Fl > 0)

    return loss, grad

def _compute_reprs(net_in, net, layers_style, layers_content, gram_scale = 1):

    """
        Compute representation matrices for an image.
    """

    # input data and forward pass
    (repr_s, repr_c) = ({}, {})
    net.blobs["data"].data[0] = net_in
    net.forward()

    # loop through combined set of layers
    # | is the set union operator
    for layer in set(layers_style) | set(layers_content):
        F = net.blobs[layer].data[0].copy()
        F.shape = (F.shape[0], -1)
        repr_c[layer] = F
        if layer in layers_style:
            repr_s[layer] = sgemm(gram_scale, F, F.T)

    return repr_s, repr_c

def style_optfn(x, net, weights, layers, reprs, ratio):

    """
        Style transfer optimization callback for scipy.optimize.minimize().

        :param numpy.ndarray x:
            Flattened data array.

        :param caffe.Net net:
            Network to use to generate gradients.

        :param dict weights:
            Weights to use in the network.

        :param list layers:
            Layers to use in the network.

        :param tuple reprs:
            Representation matrices packed in a tuple.

        :param float ratio:
            Style-to-content ratio.
    """

    # update params
    layers_style = weights["style"].keys()
    layers_content = weights["content"].keys()
    net_in = x.reshape(net.blobs["data"].data.shape[1:])

    # compute representations
    (G_style, F_content) = reprs
    (G, F) = _compute_reprs(net_in, net, layers_style, layers_content)

    # backprop by layer
    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    for i, layer in enumerate(reversed(layers)):
        next_layer = None if i == len(layers) - 1 else layers[-i - 2]
        grad = net.blobs[layer].diff[0]

        # style contribution
        if layer in layers_style:
            wl = weights["style"][layer]
            (l, g) = _compute_style_grad(F, G, G_style, layer)
            loss += wl * l * ratio
            grad += wl * g.reshape(grad.shape) * ratio

        # content contribution
        if layer in layers_content:
            wl = weights["content"][layer]
            (l, g) = _compute_content_grad(F, F_content, layer)
            loss += wl * l
            grad += wl * g.reshape(grad.shape)

        # compute gradient
        net.backward(start = layer, end = next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    # format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)

    return loss, grad

class StyleTransfer(object):

    """
        The class for Style transfer.
    """

    def __init__(self, model_name):
        
        """
            Initialize the model used for style transfer.

            :param str model_name:
                Model to use.
        """

        # __file__ is the file name of the file you run in Python
        # os.path.abspath gets the absolute path of the param
        style_path = os.path.abspath(os.path.split(__file__)[0])
        # os.path.join join the path segments
        base_path = os.path.join(style_path, "models", model_name)

        # vgg19
        if model_name == "vgg19":
            model_file = os.path.join(base_path, "VGG_ILSVRC_19_layers_deploy.prototxt")
            pretrained_file = os.path.join(base_path, "VGG_ILSVRC_19_layers.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = VGG19_WEIGHTS

        # vgg16
        elif model_name == "vgg16":
            model_file = os.path.join(base_path, "VGG_ILSVRC_16_layers_deploy.prototxt")
            pretrained_file = os.path.join(base_path, "VGG_ILSVRC_16_layers.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = VGG16_WEIGHTS

        # googlenet
        elif model_name == "googlenet":
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "googlenet_style.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = GOOGLENET_WEIGHTS

        # caffenet
        elif model_name == "caffenet":
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "bvlc_reference_caffenet.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = CAFFENET_WEIGHTS

        else:
            assert False, "model not available"

        # add model and weights
        self.load_model(model_file, pretrained_file, mean_file)
        self.weights = weights.copy()

        # get the style and content layers
        self.layers = []
        for layer in self.net.blobs:
            if layer in self.weights["style"] or layer in self.weights["content"]:
                self.layers.append(layer)

    def load_model(self, model_file, pretrained_file, mean_file):

        """
            Loads specified model from caffe install (see caffe docs).

            :param str model_file:
                Path to model protobuf.

            :param str pretrained_file:
                Path to pretrained caffe model.

            :param str mean_file:
                Path to mean file.
        """

        # load net
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        
        # all models used are trained on imagenet data
        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        # subtract the mean image from the data
        transformer.set_mean("data", np.load(mean_file).mean(1).mean(1))
        # from RGB to BGR
        transformer.set_channel_swap("data", (2, 1, 0))
        # from (H, W, C) to (C, H, W)
        transformer.set_transpose("data", (2, 0, 1))
        # input blob = input * scale
        transformer.set_raw_scale("data", 255)

        # add net parameters
        self.net = net
        self.transformer = transformer

    def get_generated(self):

        """
            Saves the generated image (net input, after optimization).

            :param str path:
                Output path.
        """

        # the generated image is in data blob
        data = self.net.blobs["data"].data
        img_out = self.transformer.deprocess("data", data)
        return img_out
    
    def _rescale_net(self, img):
        
        """
            Rescale the network to fit a particular image.
        """

        # get new dimensions and rescale net + transformer
        new_dims = (1, img.shape[2]) + img.shape[:2]
        self.net.blobs["data"].reshape(*new_dims)
        self.transformer.inputs["data"] = new_dims

    def _make_noise_input(self, init):

        """
            Create an initial input (generated) image.
        """

        # specify dimensions and create grid in Fourier domain
        dims = tuple(self.net.blobs["data"].data.shape[2:]) + \
               (self.net.blobs["data"].data.shape[1], )
        grid = np.mgrid[0:dims[0], 0:dims[1]]

        # create frequency representation for pink noise
        Sf = (grid[0] - (dims[0] - 1) / 2.0)**2 + \
             (grid[1] - (dims[1] - 1) / 2.0)**2
        Sf[np.where(Sf == 0)] = 1
        Sf = np.sqrt(Sf)
        Sf = np.dstack((Sf**int(init), ) * dims[2])

        # apply ifft to create pink noise and normalize
        ifft_kernel = np.cos(2 * np.pi * np.random.randn(*dims)) + \
                      1j * np.sin(2 * np.pi * np.random.randn(*dims))
        img_noise = np.abs(ifftn(Sf * ifft_kernel))
        img_noise -= img_noise.min()
        img_noise /= img_noise.max()

        # preprocess the pink noise image
        x0 = self.transformer.preprocess("data", img_noise)

        return x0

    def transfer_style(self, img_style, img_content, length = 512, ratio = 1e5,
                       n_iter = 512, init = "-1", verbose = False, callback = None):
        
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in floating point, RGB format.

            :param function callback:
                A callback function, which takes images at iterations.
        """

        # assume that ConvNet input is square
        orig_dim = min(self.net.blobs["data"].shape[2:])

        # rescale the images
        scale = max(length / float(max(img_style.shape[:2])),
                    orig_dim / float(min(img_style.shape[:2])))
        img_style = rescale(img_style, STYLE_SCALE * scale)
        scale = max(length / float(max(img_content.shape[:2])),
                    orig_dim / float(min(img_content.shape[:2])))
        img_content = rescale(img_content, scale)

        # compute style representations
        self._rescale_net(img_style)
        layers = self.weights["style"].keys()
        net_in = self.transformer.preprocess("data", img_style)
        G_style = _compute_reprs(net_in, self.net, layers, [], gram_scale = 1)[0]

        # compute content representations
        self._rescale_net(img_content)
        layers = self.weights["content"].keys()
        net_in = self.transformer.preprocess("data", img_content)
        F_content = _compute_reprs(net_in, self.net, [], layers)[1]

        # generate initial net input
        # "content" = content image, see kaishengtai/neuralart
        if isinstance(init, np.ndarray):
            img0 = self.transformer.preprocess("data", init)
        elif init == "content":
            img0 = self.transformer.preprocess("data", img_content)
        elif init == "mixed":
            img0 = 0.95*self.transformer.preprocess("data", img_content) + \
                   0.05*self.transformer.preprocess("data", img_style)
        else:
            img0 = self._make_noise_input(init)

        # the following code is for the L-BFGS optimization configurations
        # compute data bounds
        data_min = -self.transformer.mean["data"][:,0,0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])] * (img0.size / 3) + \
                      [(data_min[1], data_max[1])] * (img0.size / 3) + \
                      [(data_min[2], data_max[2])] * (img0.size / 3)

        # optimization params
        grad_method = "L-BFGS-B"
        reprs = (G_style, F_content)
        minfn_args = {
            "args": (self.net, self.weights, self.layers, reprs, ratio),
            "method": grad_method, "jac": True, "bounds": data_bounds,
            "options": {"maxcor": 8, "maxiter": n_iter, "disp": verbose}
        }

        # optimize
        minfn_args["callback"] = callback
        res = minimize(style_optfn, img0.flatten(), **minfn_args).nit

        return res

def main(args):
    """
        Entry point.
    """

    # set GPU/CPU mode
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    # load images
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    
    # artistic style class
    st = StyleTransfer(args.model.lower())

    # perform style transfer
    start = timeit.default_timer()
    n_iters = st.transfer_style(img_style, img_content, length = args.length, 
                                init = args.init, ratio = np.float(args.ratio), 
                                n_iter = args.num_iters, verbose = args.verbose)
    end = timeit.default_timer()
    print("Ran {0} iters in {1:.0f}s.".format(n_iters, end - start))
    img_out = st.get_generated()

    # output path
    if args.output is not None:
        out_path = args.output
    else:
        out_path_fmt = (os.path.splitext(os.path.split(args.content_img)[1])[0], 
                        os.path.splitext(os.path.split(args.style_img)[1])[0], 
                        args.model, args.init, args.ratio, args.num_iters)
        # os.path.split extracts the file name without the entire path
        # os.path.splitext discards the extension in the file name obtained by *.split
        out_path = "outputs/{0}-{1}-{2}-{3}-{4}-{5}.jpg".format(*out_path_fmt)

    # save the output image
    imsave(out_path, img_as_ubyte(img_out))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


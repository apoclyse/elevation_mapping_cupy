#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp


def get_filter_torch(w1, w2, w3, w_out, **kwargs):
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage

    class TraversabilityFilterCuPy:
        def __init__(self, w1, w2, w3, w_out):
            self.w1 = cp.asarray(w1, dtype=cp.float32)
            self.w2 = cp.asarray(w2, dtype=cp.float32)
            self.w3 = cp.asarray(w3, dtype=cp.float32)
            self.w_out_flat = cp.asarray(w_out[0, :, 0, 0], dtype=cp.float32)

            # Pre-prepare kernels for efficiency
            self.kernels1 = [self.w1[c, 0] for c in range(4)]

            self.kernels2 = []
            for c in range(4):
                k3 = self.w2[c, 0]
                k5 = cp.zeros((5, 5), dtype=cp.float32)
                for i in range(3):
                    for j in range(3):
                        k5[i*2, j*2] = k3[i, j]
                self.kernels2.append(k5)

            self.kernels3 = []
            for c in range(4):
                k3 = self.w3[c, 0]
                k7 = cp.zeros((7, 7), dtype=cp.float32)
                for i in range(3):
                    for j in range(3):
                        k7[i*3, j*3] = k3[i, j]
                self.kernels3.append(k7)

        def __call__(self, elevation_cupy):
            # Input is (202, 202) CuPy array
            H, W = elevation_cupy.shape
            out1 = cp.zeros((4, H, W), dtype=cp.float32)
            out2 = cp.zeros((4, H, W), dtype=cp.float32)
            out3 = cp.zeros((4, H, W), dtype=cp.float32)

            # Conv1
            for c in range(4):
                out1[c] = ndimage.correlate(elevation_cupy, self.kernels1[c], mode='constant', cval=0.0)

            # Conv2 (dilated 2)
            for c in range(4):
                out2[c] = ndimage.correlate(elevation_cupy, self.kernels2[c], mode='constant', cval=0.0)

            # Conv3 (dilated 3)
            for c in range(4):
                out3[c] = ndimage.correlate(elevation_cupy, self.kernels3[c], mode='constant', cval=0.0)

            # Crop to valid region (equivalent to PyTorch padding=0 / crop)
            out1_cropped = out1[:, 3:-3, 3:-3]
            out2_cropped = out2[:, 3:-3, 3:-3]
            out3_cropped = out3[:, 3:-3, 3:-3]

            # Concatenate along channel axis
            out = cp.concatenate((out1_cropped, out2_cropped, out3_cropped), axis=0)
            
            # Apply absolute activation before final 1x1 convolution
            out = cp.abs(out)

            # 1x1 convolution output
            out_conv = cp.sum(out * self.w_out_flat[:, cp.newaxis, cp.newaxis], axis=0)
            
            # Exp activation
            out_conv = cp.exp(-out_conv)

            return out_conv[cp.newaxis, cp.newaxis, :, :]

        def cuda(self):
            return self

        def eval(self):
            return self

    return TraversabilityFilterCuPy(w1, w2, w3, w_out)


def get_filter_chainer(*args, **kwargs):
    import os

    os.environ["CHAINER_WARN_VERSION_MISMATCH"] = "0"
    import chainer
    import chainer.links as L
    import chainer.functions as F

    class TraversabilityFilter(chainer.Chain):
        def __init__(self, w1, w2, w3, w_out, use_cupy=True):
            super(TraversabilityFilter, self).__init__()
            self.conv1 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=1, nobias=True, initialW=w1)
            self.conv2 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=2, nobias=True, initialW=w2)
            self.conv3 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=3, nobias=True, initialW=w3)
            self.conv_out = L.Convolution2D(12, 1, ksize=1, nobias=True, initialW=w_out)

            if use_cupy:
                self.conv1.to_gpu()
                self.conv2.to_gpu()
                self.conv3.to_gpu()
                self.conv_out.to_gpu()
            chainer.config.train = False
            chainer.config.enable_backprop = False

        def __call__(self, elevation):
            out1 = self.conv1(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
            out2 = self.conv2(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
            out3 = self.conv3(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = F.concat((out1, out2, out3), axis=1)
            out = self.conv_out(F.absolute(out))
            return F.exp(-out).array

    traversability_filter = TraversabilityFilter(*args, **kwargs)
    return traversability_filter


if __name__ == "__main__":
    import cupy as cp
    from parameter import Parameter

    elevation = cp.random.randn(202, 202, dtype=cp.float32)
    print("elevation ", elevation.shape)
    param = Parameter()
    fc = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
    print("chainer ", fc(elevation))

    ft = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
    print("torch ", ft(elevation))

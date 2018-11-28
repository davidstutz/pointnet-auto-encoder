# PointNet Auto Encoder

This repository contains a Torch implementation of a PointNet Auto Encoder,
inspired by [1] and [2].

    [1] Charles Ruizhongtai Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas:
        PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CoRR abs/1612.00593 (2016)
    [2] Haoqiang Fan, Hao Su, Leonidas J. Guibas:
        A Point Set Generation Network for 3D Object Reconstruction from a Single Image. CoRR abs/1612.00603 (2016)

If you use this code, please also cite the following master thesis:

    @misc{Stutz2017,
        author = {David Stutz},
        title = {Learning Shape Completion from Bounding Boxes with CAD Shape Priors},
        month = {September},
        year = {2017},
        institution = {RWTH Aachen University},
        address = {Aachen, Germany},
        howpublished = {http://davidstutz.de/},
    }

![Illustration of results.](screenshot.png?raw=true "Illustration of results.")

## Installation

First of all, make sure to have Torch installed, for example through
[torch/distro](https://github.com/torch/distro) which includes the required
`(cu)nn(x)` packages. Then, the C++ code can be compiled using

    # CPU code
    cd lib/cpp/cpu
    mkdir build
    cd build
    cmake ..
    make
    # GPU code
    cd ..
    cd gpu/
    mkdir build
    cd build
    cmake ..
    make

Both the CPU and GPU code can be tested by running the following tests:
    
    # within the build directory
    ./tests/test_chamfer_distance
    ./tests/test_max_distance

For the GPU code, you need to have CUDA installed, recommended is CUDA 8.
However, it also runs with lower CUDA version when adapting the used architecture.
For CUDA 8, using a Tesla K40, the compute architecture is `sm_35`
as shown in `lib/gpu/CMakeLists.txt`:

    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-O2;-DVERBOSE")

If you use a different CUDA version and/or graphics card, make sure to
adapt the architecture accordingly. Then rerun the tests to see if it works.
When you still get errors such as

    CUDA error at /BS/dstutz/work/shape-completion/code/release/pointnet_auto_encoder/lib/cpp/gpu/chamfer_distance.cu:80 code=30(cudaErrorUnknown) "cudaMalloc(&d_loss, sizeof(float))" 
    CUDA error at /BS/dstutz/work/shape-completion/code/release/pointnet_auto_encoder/lib/cpp/gpu/chamfer_distance.cu:81 code=30(cudaErrorUnknown) "cudaMemcpy(d_loss, &loss, sizeof(float), cudaMemcpyHostToDevice)" 
    CUDA error at /BS/dstutz/work/shape-completion/code/release/pointnet_auto_encoder/lib/cpp/gpu/chamfer_distance.cu:90 code=30(cudaErrorUnknown) "cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost)" 
    CUDA error at /BS/dstutz/work/shape-completion/code/release/pointnet_auto_encoder/lib/cpp/gpu/chamfer_distance.cu:92 code=30(cudaErrorUnknown) "cudaFree(d_loss)"

it is very likely that the set architecture does not meet your installed CUDA version!

For training the auto encoder, the following Torch packages are required in
addition to torch/distro:

* [json](https://github.com/harningt/luajson)
* [hdf5](https://github.com/deepmind/torch-hdf5)
* [lfs](http://keplerproject.github.io/luafilesystem)

Follow the instructions from the respective packages.

## Usage

A usage example is provided in `auto_encoder_train.lua` which includes
three different models and a simple training and evaluation loop. Also see
the corresponding blog article on [davidstutz.de](http://davidstutz.de/).

## License

License for source code corresponding to:

D. Stutz. **Learning Shape Completion from Bounding Boxes with CAD Shape Priors.** Master Thesis, RWTH Aachen University, 2017.

Copyright (c) 2018 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.

# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import tkinter as tk
import numpy as np
import time
from contextlib import contextmanager
import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import gloo, gl
from pyopengltk import OpenGLFrame
import torch
from torch.autograd import Variable

# TkInter widget that can draw torch tensors directly from GPU memory

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()

def create_shared_texture(w, h, c=4,
        map_flags=graphics_map_flags.WRITE_DISCARD,
        dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
    tex.activate() # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(
        int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer

# Shape batch as square if possible
def get_grid_dims(B):
    S = int(B**0.5 + 0.5)
    while B % S != 0:
        S -= 1
    return (B // S, S)

def create_gl_texture(tensor_shape):
    if len(tensor_shape) != 4:
        raise RuntimeError('Please provide a tensor of shape NCHW')
    
    N, C, H, W = tensor_shape

    cols, rows = get_grid_dims(N)
    tex, cuda_buffer = create_shared_texture(W*cols, H*rows, 4)

    return tex, cuda_buffer

# Create window with OpenGL context
class TorchImageView(OpenGLFrame):
    def __init__(self, root = None, show_fps=True, **kwargs):
        self.root = root or tk.Tk()
        self.width = kwargs.get('width', 512)
        self.height = kwargs.get('height', 512)
        self.show_fps = show_fps
        self.pycuda_initialized = False
        self.animate = 0 # disable internal main loop
        OpenGLFrame.__init__(self, root, **kwargs)

    # Called by pyopengltk.BaseOpenGLFrame
    # when the frame goes onto the screen
    def initgl(self):
        if not self.pycuda_initialized:
            self.setup_gl(self.width, self.height)
            self.pycuda_initialized = True
        
        """Initalize gl states when the frame is created"""
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.dt_history = [1000/60]
        self.t0 = time.time()
        self.t_last = self.t0
        self.nframes = 0

    def setup_gl(self, width, height):
        # setup pycuda and torch
        import pycuda.gl.autoinit
        import pycuda.gl

        assert torch.cuda.is_available(), "PyTorch: CUDA is not available"
        print('Using GPU {}'.format(torch.cuda.current_device()))
        
        # Create tensor to be shared between GL and CUDA
        # Always overwritten so no sharing is necessary
        dummy = torch.cuda.FloatTensor((1))
        dummy.uniform_()
        dummy = Variable(dummy)
        
        # Create a buffer with pycuda and gloo views, using tensor created above
        self.tex, self.cuda_buffer = create_gl_texture((1, 3, width, height))
        
        # create a shader to program to draw to the screen
        vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
        fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } """
        # Build the program and corresponding buffers (with 4 vertices)
        self.screen = gloo.Program(vertex, fragment, count=4)
        
        # NDC coordinates:         Texcoords:          Vertex order,
        # (-1, +1)       (+1, +1)   (0,0)     (1,0)    triangle strip:
        #        +-------+               +----+          1----3
        #        |  NDC  |               |    |          |  / | 
        #        | SPACE |               |    |          | /  |
        #        +-------+               +----+          2----4
        # (-1, -1)       (+1, -1)   (0,1)     (1,1)
        
        # Upload data to GPU
        self.screen['position'] = [(-1,+1), (-1,-1), (+1,+1), (+1,-1)]
        self.screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
        self.screen['scale'] = 1.0
        self.screen['tex'] = self.tex

    # Don't call directly, use update() instead
    def redraw(self):
        t_now = time.time()
        dt = t_now - self.t_last
        self.t_last = t_now

        self.dt_history = ([dt] + self.dt_history)[:50]
        dt_mean = sum(self.dt_history) / len(self.dt_history)

        if self.show_fps and self.nframes % 60 == 0:
            self.master.title('FPS: {:.0f}'.format(1 / dt_mean))

    def draw(self, img):
        assert len(img.shape) == 4, "Please provide an NCHW image tensor"
        assert img.device.type == "cuda", "Please provide a CUDA tensor"

        if img.dtype.is_floating_point:
            img = (255*img).byte()
        
        # Tile images
        N, C, H, W = img.shape

        if N > 1:
            cols, rows = get_grid_dims(N)
            img = img.reshape(cols, rows, C, H, W)
            img = img.permute(2, 1, 3, 0, 4) # [C, rows, H, cols, W]
            img = img.reshape(1, C, rows*H, cols*W)

        tensor = img.squeeze().permute(1, 2, 0).data # CHW => HWC
        if C == 3:
            tensor = torch.cat((tensor, tensor[:,:,0:1]),2) # add the alpha channel
            tensor[:,:,3] = 1 # set alpha
        
        tensor = tensor.contiguous()

        tex_h, tex_w, _ = self.tex.shape
        tensor_h, tensor_w, _ = tensor.shape

        if (tex_h, tex_w) != (tensor_h, tensor_w):
            print(f'Resizing texture to {tensor_w}*{tensor_h}')
            self.tex, self.cuda_buffer = create_gl_texture((N, C, H, W)) # original shape
            self.screen['tex'] = self.tex

        # copy from torch into buffer
        assert self.tex.nbytes == tensor.numel()*tensor.element_size(), "Tensor and texture shape mismatch!"
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = self.tex.nbytes//tensor_h
            cpy.height = tensor_h
            cpy(aligned=False)
            torch.cuda.synchronize()
        
        # draw to screen
        self.screen.draw(gl.GL_TRIANGLE_STRIP)

    def update(self):
        self.update_idletasks()
        self.tkMakeCurrent()
        self.redraw()
        self.tkSwapBuffers()

# USAGE:
# root = tk.Tk()
# iv = TorchImageView(root, width=512, height=512)
# iv.pack(fill='both', expand=True)
# while True:
#     iv.draw(nchw_tensor)
#     root.update()
#     iv.update()
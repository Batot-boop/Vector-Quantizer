import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def pad_image(image, block_h, block_w):
    h, w = image.shape
    pad_h = (block_h - h % block_h) % block_h
    pad_w = (block_w - w % block_w) % block_w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
    return padded, h, w

def extract_blocks(image, block_h, block_w):
    h, w = image.shape
    blocks = [image[i:i+block_h, j:j+block_w].flatten() for i in range(0, h, block_h) for j in range(0, w, block_w)]
    return np.array(blocks)

def generate_codebook(blocks, K, epsilon=1):  # Changed epsilon to 1 for 0-255 scale
    if len(blocks) == 0: return np.array([])
    dim = blocks.shape[1]
    centroid = np.mean(blocks, axis=0).astype(float)
    codebook = [centroid]
    while len(codebook) < K:
        new_codebook = []
        for c in codebook:
            new_codebook.extend([c - epsilon, c + epsilon])
        codebook = new_codebook
        for _ in range(100):
            dists = np.linalg.norm(blocks[:, np.newaxis] - np.array(codebook), axis=2)
            assignments = np.argmin(dists, axis=1)
            new_codebook = []
            changed = False
            for i in range(len(codebook)):
                assigned = blocks[assignments == i]
                new_c = np.mean(assigned, axis=0) if len(assigned) > 0 else codebook[i]
                if np.linalg.norm(new_c - codebook[i]) > 1e-3: changed = True
                new_codebook.append(new_c)
            codebook = new_codebook
            if not changed: break
    return np.round(np.array(codebook)).astype(int)

def compress(image_path, block_h, block_w, K):
    image = plt.imread(image_path)
    if len(image.shape) == 3: image = np.mean(image, axis=2)
    image = image.astype(float)
    image *= 255  # Scale to 0-255
    padded, orig_h, orig_w = pad_image(image, block_h, block_w)
    blocks = extract_blocks(padded, block_h, block_w)
    codebook = generate_codebook(blocks, K)
    dists = np.linalg.norm(blocks[:, np.newaxis] - codebook, axis=2)
    labels = np.argmin(dists, axis=1)
    dir_path = os.path.dirname(os.path.abspath(image_path))  # Get original image directory
    np.save(os.path.join(dir_path, 'codebook.npy'), codebook)
    np.save(os.path.join(dir_path, 'compressed.npy'), labels)
    np.savez(os.path.join(dir_path, 'metadata.npz'), orig_h=orig_h, orig_w=orig_w, block_h=block_h, block_w=block_w,
             pad_h=padded.shape[0], pad_w=padded.shape[1], dir_path=dir_path)
    orig_bits = orig_h * orig_w * 8
    num_blocks = len(labels)
    bits_per_label = math.ceil(math.log2(K)) if K > 0 else 0
    labels_bits = num_blocks * bits_per_label
    codebook_bits = K * (block_h * block_w) * 8
    compressed_bits = labels_bits + codebook_bits
    ratio = orig_bits / compressed_bits if compressed_bits > 0 else 0
    return ratio, dir_path

def decompress(dir_path=None):
    if dir_path is None:
        dir_path = os.getcwd()  # Fallback to current dir if not provided
    codebook = np.load(os.path.join(dir_path, 'codebook.npy'))
    labels = np.load(os.path.join(dir_path, 'compressed.npy'))
    data = np.load(os.path.join(dir_path, 'metadata.npz'))
    orig_h = data['orig_h']
    orig_w = data['orig_w']
    block_h = data['block_h']
    block_w = data['block_w']
    pad_h = data['pad_h']
    pad_w = data['pad_w']
    reconstructed_padded = np.zeros((pad_h, pad_w))
    idx = 0
    for i in range(0, pad_h, block_h):
        for j in range(0, pad_w, block_w):
            block = codebook[labels[idx]].reshape(block_h, block_w)
            reconstructed_padded[i:i+block_h, j:j+block_w] = block
            idx += 1
    reconstructed = reconstructed_padded[:orig_h, :orig_w]
    save_path = os.path.join(dir_path, 'reconstructed.png')
    print("Saving reconstructed.png to:", save_path)  # Print path for debugging
    plt.imsave(save_path, reconstructed / 255.0, cmap='gray')  # Normalize back to 0-1 for saving
    return f"Reconstructed image saved as '{save_path}'"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VQ Compressor")
        self.dir_path = None  # To store dir_path from compress
        
        tk.Label(root, text="Image Path:").grid(row=0, column=0)
        self.path_entry = tk.Entry(root)
        self.path_entry.grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse).grid(row=0, column=2)
        
        tk.Label(root, text="Block Height:").grid(row=1, column=0)
        self.block_h = tk.Entry(root)
        self.block_h.grid(row=1, column=1)
        
        tk.Label(root, text="Block Width:").grid(row=2, column=0)
        self.block_w = tk.Entry(root)
        self.block_w.grid(row=2, column=1)
        
        tk.Label(root, text="Codebook Size (K):").grid(row=3, column=0)
        self.k_entry = tk.Entry(root)
        self.k_entry.grid(row=3, column=1)
        
        tk.Button(root, text="Compress", command=self.do_compress).grid(row=4, column=0)
        tk.Button(root, text="Decompress", command=self.do_decompress).grid(row=4, column=1)
        tk.Button(root, text="Exit", command=root.quit).grid(row=4, column=2)
    
    def browse(self):
        path = filedialog.askopenfilename()
        self.path_entry.insert(0, path)
    
    def do_compress(self):
        try:
            path = self.path_entry.get()
            bh = int(self.block_h.get())
            bw = int(self.block_w.get())
            k = int(self.k_entry.get())
            ratio, self.dir_path = compress(path, bh, bw, k)
            messagebox.showinfo("Success", f"Compression ratio: {ratio:.2f}:1\nFiles saved in {self.dir_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def do_decompress(self):
        try:
            msg = decompress(self.dir_path)
            messagebox.showinfo("Success", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
#!/usr/bin/env python3
"""
generate_golden.py — Generate reference forward/backward data for the FNN kernel.

Produces a binary file "golden_data.bin" with:
  - Network weights (W1, b1, W2, b2, W3, b3)
  - Input vector
  - Expected forward output
  - Expected backward gradients (grad_W*, grad_b*, grad_input)
  - Expected loss gradient and grad_input

Usage:  python generate_golden.py [output_path]
"""

import numpy as np
import struct
import sys
from pathlib import Path


# ---- Network dimensions (must match fnn_config.cuh DefaultConfig) ----
IN_DIM  = 32
H1_DIM  = 64
H2_DIM  = 64
OUT_DIM = 4
BATCH   = 1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(y):
    """Derivative of sigmoid given the sigmoid output y."""
    return y * (1.0 - y)


class FNNReference:
    """Reference implementation matching agent_numpy.py Linear layer."""

    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)

        # Xavier init
        self.W1 = rng.randn(H1_DIM, IN_DIM).astype(np.float32) / np.sqrt(IN_DIM)
        self.b1 = rng.randn(H1_DIM, 1).astype(np.float32) * 0.01
        self.W2 = rng.randn(H2_DIM, H1_DIM).astype(np.float32) / np.sqrt(H1_DIM)
        self.b2 = rng.randn(H2_DIM, 1).astype(np.float32) * 0.01
        self.W3 = rng.randn(OUT_DIM, H2_DIM).astype(np.float32) / np.sqrt(H2_DIM)
        self.b3 = rng.randn(OUT_DIM, 1).astype(np.float32) * 0.01

        # Gradient accumulators
        self.zero_grad()

    def zero_grad(self):
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)
        self.grad_W3 = np.zeros_like(self.W3)
        self.grad_b3 = np.zeros_like(self.b3)

    def forward(self, x):
        """
        x: (IN_DIM, BATCH) array
        Returns: (OUT_DIM, BATCH)
        Caches intermediate values for backward.
        """
        self.x = x  # cache input

        # Layer 1
        z1 = np.dot(self.W1, x) + self.b1  # (H1, B)
        self.a1 = sigmoid(z1)

        # Layer 2
        z2 = np.dot(self.W2, self.a1) + self.b2  # (H2, B)
        self.a2 = sigmoid(z2)

        # Layer 3 (linear output)
        z3 = np.dot(self.W3, self.a2) + self.b3  # (OUT, B)
        self.y = z3

        return self.y

    def backward(self, loss_grad):
        """
        loss_grad: (OUT_DIM, BATCH) = dL/dy
        Returns: grad_x (IN_DIM, BATCH) = dL/dx
        Accumulates weight/bias gradients in self.grad_*.
        """
        # Layer 3 backward (linear)
        g_z3 = loss_grad  # (OUT, B)
        self.grad_W3 += np.dot(g_z3, self.a2.T)    # (OUT, H2)
        self.grad_b3 += np.sum(g_z3, axis=1, keepdims=True)
        g_a2 = np.dot(self.W3.T, g_z3)              # (H2, B)

        # Layer 2 backward (sigmoid + linear)
        g_z2 = g_a2 * sigmoid_derivative(self.a2)   # (H2, B)
        self.grad_W2 += np.dot(g_z2, self.a1.T)    # (H2, H1)
        self.grad_b2 += np.sum(g_z2, axis=1, keepdims=True)
        g_a1 = np.dot(self.W2.T, g_z2)              # (H1, B)

        # Layer 1 backward (sigmoid + linear)
        g_z1 = g_a1 * sigmoid_derivative(self.a1)   # (H1, B)
        self.grad_W1 += np.dot(g_z1, self.x.T)     # (H1, IN)
        self.grad_b1 += np.sum(g_z1, axis=1, keepdims=True)
        g_x = np.dot(self.W1.T, g_z1)               # (IN, B)

        return g_x

    def pack_weights(self):
        """Pack weights into flat arrays matching the CUDA NetworkWeights struct layout."""
        return {
            'W1': self.W1.flatten(),          # (H1 * IN,)
            'b1': self.b1.flatten(),          # (H1,)
            'W2': self.W2.flatten(),          # (H2 * H1,)
            'b2': self.b2.flatten(),          # (H2,)
            'W3': self.W3.flatten(),          # (OUT * H2,)
            'b3': self.b3.flatten(),          # (OUT,)
        }

    def pack_grads(self):
        return {
            'grad_W1': self.grad_W1.flatten(),
            'grad_b1': self.grad_b1.flatten(),
            'grad_W2': self.grad_W2.flatten(),
            'grad_b2': self.grad_b2.flatten(),
            'grad_W3': self.grad_W3.flatten(),
            'grad_b3': self.grad_b3.flatten(),
        }


def write_binary(path, data_dict, fnn, x, loss_grad, y_golden, grad_x_golden):
    """Write all golden data to a binary file."""
    with open(path, 'wb') as f:
        # Magic number + version
        f.write(b'FNN1')  # 4 bytes magic
        f.write(struct.pack('i', 1))  # version

        # Dimensions
        f.write(struct.pack('iiiii', IN_DIM, H1_DIM, H2_DIM, OUT_DIM, BATCH))

        # Write each array: num_elements (int32) + data (float32)
        def write_arr(name, arr):
            data = np.asarray(arr, dtype=np.float32)
            f.write(struct.pack('i', data.size))
            f.write(data.tobytes())
            print(f"  wrote {name}: {data.size} floats ({data.nbytes} bytes)")

        w = fnn.pack_weights()
        g = fnn.pack_grads()

        write_arr('W1', w['W1'])
        write_arr('b1', w['b1'])
        write_arr('W2', w['W2'])
        write_arr('b2', w['b2'])
        write_arr('W3', w['W3'])
        write_arr('b3', w['b3'])

        write_arr('input_x', x)
        write_arr('loss_grad', loss_grad)

        write_arr('output_y', y_golden)
        write_arr('grad_x', grad_x_golden)

        write_arr('grad_W1', g['grad_W1'])
        write_arr('grad_b1', g['grad_b1'])
        write_arr('grad_W2', g['grad_W2'])
        write_arr('grad_b2', g['grad_b2'])
        write_arr('grad_W3', g['grad_W3'])
        write_arr('grad_b3', g['grad_b3'])

    size = Path(path).stat().st_size
    print(f"\nTotal file size: {size} bytes")


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'golden_data.bin'
    print(f"Generating golden data → {out_path}")
    print(f"Network: {IN_DIM}→{H1_DIM}→{H2_DIM}→{OUT_DIM}, batch={BATCH}\n")

    rng = np.random.RandomState(1234)

    # Create reference network
    fnn = FNNReference(seed=42)

    # Generate random input
    x = rng.randn(IN_DIM, BATCH).astype(np.float32) * 0.5

    # Forward pass
    y_golden = fnn.forward(x)

    # Generate random loss gradient
    loss_grad = rng.randn(OUT_DIM, BATCH).astype(np.float32) * 0.5

    # Backward pass
    grad_x_golden = fnn.backward(loss_grad)

    # Write binary
    write_binary(out_path, {}, fnn, x, loss_grad, y_golden, grad_x_golden)

    # Also print some diagnostics
    print(f"\nDiagnostics:")
    print(f"  y range:    [{y_golden.min():.4f}, {y_golden.max():.4f}]")
    print(f"  grad_x range: [{grad_x_golden.min():.4f}, {grad_x_golden.max():.4f}]")
    print(f"  grad_W1 range: [{fnn.grad_W1.min():.4f}, {fnn.grad_W1.max():.4f}]")
    print(f"\nDone.")


if __name__ == '__main__':
    main()

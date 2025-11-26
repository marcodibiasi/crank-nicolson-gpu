# crank-nicolson-gpu
An optimized **Crank–Nicolson** solver paired with a parallel **Preconditioned Conjugate Gradient solver**.
The program takes an initial heatmap and a physical setup, and outputs n user-defined frames showing the temporal evolution of the system, with a timestep dt between frames.

The solver currently only supports square heatmaps and Dirichlet boundary conditions.

Performance measurements have been recorded and will be presented soon.

## 🔧 How to Build and Use the Solver

### 📥 1. Clone the repository

```bash
git clone https://github.com/marcodibiasi/crank-nicolson-gpu.git
cd crank-nicolson-gpu
```

### 🔨 Build the project

Make sure you have a C compiler and OpenCL installed, then run:

```bash
make
```

### ▶️ Run the solver

The solver requires one `.pgm` grayscale heatmap and the physical parameters:

```bash
./cn input.pgm dx dt alpha iterations
```

### 📄 .pgm File

The `.pgm` file is a simple text file where:

1. The first row specifies the **width** and **height** of the image.  
2. Each subsequent row contains pixel values in the range `[0, 255]`.

Few example input files can be found in the `data/input/` directory.
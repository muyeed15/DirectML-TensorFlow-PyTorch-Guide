# DirectML Setup on Windows (TensorFlow 2.10.0 & PyTorch 0.2.4.dev240815)

This guide provides detailed instructions for setting up DirectML natively on Windows for TensorFlow 2.10.0 and PyTorch 0.2.4.dev240815. DirectML is a high-performance, hardware-accelerated DirectX 12 library for machine learning. It provides GPU acceleration for common machine learning tasks across a broad range of supported hardware and drivers, including all DirectX 12-capable GPUs.

## Requirements

### Operating Systems
- **Windows 10 Version 1709** (Build 16299 or higher), 64-bit
- **Windows 11 Version 21H2** (Build 22000 or higher), 64-bit

### Supported GPUs
- **AMD**: Radeon R5/R7/R9 2xx series or newer
- **Intel**: HD Graphics 5xx or newer
- **NVIDIA**: GeForce GTX 9xx series GPU or newer

## Installation Steps

### 1. Download and Install Miniconda

Open a command prompt (ensure it is not running as Administrator) and execute the following commands to download and install Miniconda:

```cmd
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```

### 2. Set Up the Conda Environment

Open the "Anaconda Prompt (miniconda3)" and create and activate a new Conda environment named `directml` with Python 3.10.14:

```cmd
conda create --name "directml" python==3.10.14
conda activate "directml"
```

### 3. Install numpy==1.26.4

In the activated Conda environment, install numpy==1.26.4 using pip:

```cmd
pip install numpy==1.26.4
```

### 4. Install tensorflow-cpu==2.10.0 and tensorflow-directml-plugin==0.4.0.dev230202

Now, install tensorflow-cpu==2.10.0 and tensorflow-directml-plugin==0.4.0.dev230202 using pip in the same Conda environment:

```cmd
pip install tensorflow-cpu==2.10.0
pip install tensorflow-directml-plugin==0.4.0.dev230202
```

### 5. Install torch-directml==0.2.4.dev240815

Similarly, install torch-directml==0.2.4.dev240815:

```cmd
pip install torch-directml==0.2.4.dev240815
```

## Usage

### TensorFlow with DirectML

To verify that TensorFlow is working correctly with DirectML, run the following Python script:

```python
import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
```

### PyTorch with DirectML

To verify that PyTorch is working with DirectML, run the following Python script:

```python
import torch
import torch_directml

dml = torch_directml.device()

tensor1 = torch.tensor([1]).to(dml)
tensor2 = torch.tensor([2]).to(dml)

dml_algebra = tensor1 + tensor2
print(dml_algebra.item())
```

### Monitoring GPU Usage

While running your DirectML-based scripts, you can monitor GPU usage using the Task Manager:

1. **Open Task Manager**: Press `Ctrl + Shift + Esc`.
2. **Navigate to the Performance Tab**: Click on the "Performance" tab.
3. **Select GPU**: In the left sidebar, select your GPU.
4. **Change the View**: In the GPU tab, change one of the graphs to "Compute 0", "Compute 1", etc., to monitor GPU computation usage.
5. **Check Dedicated GPU Memory**: Observe "Dedicated GPU Memory" usage to see how much memory your TensorFlow or PyTorch tasks are using.

## Performance Comparison

To compare the performance of DirectML with GPU acceleration versus CPU execution, you can use the following scripts. These scripts perform intensive matrix operations and measure the time taken on DirectML (if available) versus CPU.

### PyTorch Performance Comparison

This script performs complex matrix operations and compares the execution time on DirectML (if available) with that on the CPU:

```python
import torch
import torch_directml
import time

# Function to perform a complex operation
def complex_operations(tensor):
    # Perform a series of intensive operations
    result_mm = torch.mm(tensor, tensor)
    result_elementwise = tensor * tensor
    result_sum = torch.sum(result_elementwise)
    return result_mm, result_elementwise, result_sum

# Check if DirectML is available
dml = torch_directml.device()
device = dml if dml else 'cpu'

# Define a very large size for the matrices
N = 10000  # Adjust this size based on your system's capabilities

# Create random tensors
tensor1 = torch.rand(N, N, device=device)
tensor2 = torch.rand(N, N, device=device)

# Time the DirectML operation if available
if device != 'cpu':
    start_time = time.time()
    
    # Perform complex operations on DirectML
    result_mm_dml, result_elementwise_dml, result_sum_dml = complex_operations(tensor1)
    
    dml_time = time.time() - start_time
else:
    dml_time = None

# Move tensors to CPU for CPU operation
tensor1_cpu = tensor1.cpu()
tensor2_cpu = tensor2.cpu()

# Time the CPU operation
start_time = time.time()

# Perform the same complex operations on CPU
result_mm_cpu, result_elementwise_cpu, result_sum_cpu = complex_operations(tensor1_cpu)

cpu_time = time.time() - start_time

# Print the results
if dml_time is not None:
    print(f"Time taken on DirectML (GPU): {dml_time:.4f} seconds")
    print(f"DirectML Result Sum: {result_sum_dml.item():.4f}")
    # To avoid overwhelming the output, only show the shape of the results
    print(f"DirectML Result MM Shape: {result_mm_dml.shape}")
    print(f"DirectML Result Elementwise Shape: {result_elementwise_dml.shape}")

print(f"Time taken on CPU: {cpu_time:.4f} seconds")
print(f"CPU Result Sum: {result_sum_cpu.item():.4f}")
# To avoid overwhelming the output, only show the shape of the results
print(f"CPU Result MM Shape: {result_mm_cpu.shape}")
print(f"CPU Result Elementwise Shape: {result_elementwise_cpu.shape}")
```

## Notes

- The specific version numbers provided are tailored to this setup and might change in the future. Ensure compatibility if you use newer versions of TensorFlow, PyTorch, or DirectML.
- Always verify your hardware compatibility, especially if using older or newer GPUs.

This setup will allow you to leverage GPU acceleration on Windows with TensorFlow and PyTorch via DirectML, enabling high-performance machine learning tasks on a wide range of hardware.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This guide was created with insights from Microsoft’s DirectML documentation and resources.

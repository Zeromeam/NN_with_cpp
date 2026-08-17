# Neural Network in C++

A feed-forward neural network project for classifying handwritten digits from
MNIST-style CSV data. The application demonstrates the full classifier workflow
in C++: input preprocessing, target encoding, training, inference, and a simple
Windows console visualization.

## Architecture

The network is configured in `ConsoleApplication6.cpp` as:

```cpp
neuralnetwork n(784, 100, 10, 0.3);
```

- 784 input values for a 28 × 28 grayscale image
- one hidden layer with 100 units
- 10 output scores, one for each digit
- a learning-rate argument of 0.3

## Data pipeline

Each CSV row begins with the digit label followed by 784 pixel values. The
application:

1. reads and parses the CSV rows;
2. scales pixel values from 0–255 into 0.01–1.00;
3. encodes the expected class as 0.99 and the remaining classes as 0.01;
4. trains the network over the prepared samples; and
5. prints the ten output scores for each queried image.

The input scaling used in the source is:

```text
normalized(x) = (x / 255) * 0.99 + 0.01
```

## Repository contents

- `NN_with_cpp/ConsoleApplication6/ConsoleApplication6.cpp` — application entry
  point and training workflow
- `NN_with_cpp/ConsoleApplication6/ConsoleApplication6.vcxproj` — Visual Studio
  project configuration
- `NN_with_cpp/ConsoleApplication6/111.csv` — labelled sample data used by the
  project

The application targets Windows and uses the Win32 console drawing API to render
the 28 × 28 input at a larger scale.

Explore the architecture, hidden activations, output scores, and learned
receptive fields in the
[portfolio case study](https://medoali.at/work/neural-network-cpp).

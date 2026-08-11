# Neural Network in C++

An early from-scratch feed-forward classifier for handwritten MNIST digits. The
retained entry point makes the data contract and network shape explicit:

    neuralnetwork n(784, 100, 10, 0.3);

That is a 784-value image input, one 100-unit hidden layer, ten digit outputs,
and a learning-rate argument of 0.3.

## What the retained code shows

ConsoleApplication6.cpp documents the end-to-end application flow:

1. Parse label-first MNIST CSV rows.
2. Normalize every pixel from 0–255 into 0.01–1.00.
3. Encode targets as 0.99 for the correct digit and 0.01 elsewhere.
4. Train the 784 → 100 → 10 network over the loaded rows.
5. Query the model and print all ten output scores.
6. Draw a 28 × 28 input in a Windows console window for inspection.

The formula used by the source for each input pixel x is:

    normalized(x) = (x / 255) * 0.99 + 0.01

## Honest source status

This repository is incomplete as a standalone build. The Visual Studio entry
point and a labelled MNIST CSV are present, but the historical neuralnetwork.h
and Matrix implementation referenced by the program were not retained. Compiled
Debug artifacts are included for provenance, but they do not replace the missing
source and should not be treated as a reproducible release.

The portfolio therefore does not pretend to run the lost original model. It
contains a clearly labelled deterministic reconstruction that matches the
verified 784 → 100 → 10 architecture, normalization, and target encoding. The
reconstruction was trained offline, evaluated on a held-out split, quantized,
and shipped as static browser data.

## Limitations

- The original header implementations and trained weights are missing.
- The retained application is Windows-specific and is not currently buildable.
- The CSV appears to be a compact project copy, not a documented canonical MNIST
  distribution.
- The reconstructed weights demonstrate the architecture; they are not claimed
  to be the original experiment's parameters or result.

Inspect the code-to-math explanation, hidden activations, output scores, and
learned receptive fields in the
[portfolio case study](https://medoali.at/work/neural-network-cpp).

# Graphcore tutorials

These tutorials provide hands-on programming exercises to enable you to familiarise yourself with creating, running and profiling programs on the IPU. They are part of the Developer resources provided by Graphcore: https://www.graphcore.ai/developer.

Each of the tutorials contains its own README file with full instructions.

## Poplar

Poplar is the underlying C++ framework for developing and executing code on the Graphcore IPU.
It provides a generic interface on which the other frameworks are built.

- [Tutorial 1: Programs and variables](poplar/tut1_variables)
- [Tutorial 2: Using PopLibs](poplar/tut2_operations)
- [Tutorial 3: Writing vertex code](poplar/tut3_vertices)
- [Tutorial 4: Profiling output](poplar/tut4_profiling)
- [Tutorial 5: Matrix-vector multiplication](poplar/tut5_matrix_vector)
- [Tutorial 6: Matrix-vector multiplication optimisation](poplar/tut6_matrix_vector_opt)

## PyTorch

Getting started with the IPU:

- [Starter tutorial: MNIST training example](../simple_applications/pytorch/mnist)
- [From 0 to 1: Introduction to PopTorch](pytorch/basics)

Exchanging data between the host and the IPU:

- [Efficient data loading with PopTorch](pytorch/efficient_data_loading)

Maximising compute on the IPU:

- [Half and mixed precision in PopTorch](pytorch/mixed_precision)

Using multiple IPUs and handling large models:

- [Parallel execution using pipelining](pytorch/pipelining)
- [Fine-tuning BERT with HuggingFace and PopTorch](pytorch/finetuning_bert)

Debugging and analysis:

- [Observing tensors in PopTorch](pytorch/observing_tensors)

## PopVision

PopVision is Graphcore's suite of graphical application analysis tools.

- [Instrumenting applications and using the PopVision System Analyser](popvision/system_analyser_instrumentation)
- [Accessing profiling information with libpva](popvision/libpva)
- Profiling output with the PopVision Graph Analyser is currently included in [Poplar Tutorial 4: profiling output](poplar/tut4_profiling)

## PopXL and popxl.addons

PopXL and popxl.addons are Graphcore frameworks which provide low level
control of the IPU through an expressive
Python interface designed for machine learning applications.

- [Tutorial 1: Basic Concepts](popxl/1_basic_concepts)
- [Tutorial 2: Custom Optimiser](popxl/2_custom_optimiser)

Improving performance and optimising throughput:

- [Tutorial 3: Data Parallelism](popxl/3_data_parallelism)
- [Tutorial 4: Pipelining](popxl/4_pipelining)
- [Tutorial 5: Remote Variables](popxl/5_remote_variables)
- [Tutorial 6: Phased Execution](popxl/6_phased_execution)

## Standard tools

In this folder you will find explanations of how to use standard deep learning tools
with the Graphcore IPU. Guides included are:

- [Using IPUs from Jupyter Notebooks](standard_tools/using_jupyter)

# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```bash
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1 & Task 3.2 Parallel Check Output:

The terminal command used to run the parallel analytics script parallel_check.py provided in the repository is:

```bash
python project/parallel_check.py
```

The diagnostics output of the parallel analytics script from the terminal is as follows:

```console
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/ste
phen_shen/Desktop/ML_Engineering/workspace/mod3-
Stephen0512/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/stephen_shen/Desktop/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (164)
-------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                        |
        out: Storage,                                                                |
        out_shape: Shape,                                                            |
        out_strides: Strides,                                                        |
        in_storage: Storage,                                                         |
        in_shape: Shape,                                                             |
        in_strides: Strides,                                                         |
    ) -> None:                                                                       |
        # Check if the tensors are stride-aligned                                    |
        if (                                                                         |
            len(out_strides) == len(in_strides)                                      |
            and np.array_equal(out_strides, in_strides)                              |
            and np.array_equal(out_shape, in_shape)                                  |
        ):                                                                           |
            # Fast path: tensors are stride-aligned, avoid indexing                  |
            for i in prange(out.size):-----------------------------------------------| #0
                out[i] = fn(in_storage[i])                                           |
            return                                                                   |
                                                                                     |
        # Slow path: tensors are not stride-aligned                                  |
                                                                                     |
        # Process each element of the output tensor in parallel                      |
        for i in prange(out.size):---------------------------------------------------| #1
            # Initialize index arrays for input and output tensors                   |
            out_index = np.empty(MAX_DIMS, np.int32)  # Output tensor index          |
            in_index = np.empty(MAX_DIMS, np.int32)   # Input tensor index           |
                                                                                     |
            # Convert flat index i to tensor indices for output tensor               |
            to_index(i, out_shape, out_index)                                        |
                                                                                     |
            # Handle broadcasting between tensors to get input tensor index          |
            broadcast_index(out_index, out_shape, in_shape, in_index)                |
                                                                                     |
            # Convert indices to positions in storage                                |
            in_pos = index_to_position(in_index, in_strides)   # Input position      |
            out_pos = index_to_position(out_index, out_strides) # Output position    |
                                                                                     |
            # Apply function and store result                                        |
            out[out_pos] = fn(in_storage[in_pos])                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/stephen_shen/Deskto
p/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (188) is
hoisted out of the parallel loop labelled #1 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)  # Output tensor index
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/stephen_shen/Deskto
p/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (189) is
hoisted out of the parallel loop labelled #1 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)   # Input tensor index
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/ste
phen_shen/Desktop/ML_Engineering/workspace/mod3-
Stephen0512/minitorch/fast_ops.py (230)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/stephen_shen/Desktop/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (230)
-----------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                            |
        out: Storage,                                                                    |
        out_shape: Shape,                                                                |
        out_strides: Strides,                                                            |
        a_storage: Storage,                                                              |
        a_shape: Shape,                                                                  |
        a_strides: Strides,                                                              |
        b_storage: Storage,                                                              |
        b_shape: Shape,                                                                  |
        b_strides: Strides,                                                              |
    ) -> None:                                                                           |
        # Check if the tensors are stride-aligned                                        |
        if (                                                                             |
            len(out_strides) == len(a_strides) == len(b_strides)                         |
            and np.array_equal(out_strides, a_strides)                                   |
            and np.array_equal(out_strides, b_strides)                                   |
            and np.array_equal(out_shape, a_shape)                                       |
            and np.array_equal(out_shape, b_shape)                                       |
        ):                                                                               |
            # Fast path: tensors are stride-aligned, avoid indexing                      |
            for i in prange(out.size):---------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                  |
            return                                                                       |
                                                                                         |
        # Slow path: tensors are not stride-aligned                                      |
                                                                                         |
        # Process each element in the output tensor in parallel                          |
        for i in prange(out.size):-------------------------------------------------------| #3
                                                                                         |
            # Initialize index arrays for input and output tensor indices                |
            out_index = np.empty(MAX_DIMS, np.int32)  # Output tensor index              |
            a_index = np.empty(MAX_DIMS, np.int32)    # First input tensor index         |
            b_index = np.empty(MAX_DIMS, np.int32)    # Second input tensor index        |
                                                                                         |
            # Convert flat index i to tensor indices for output tensor                   |
            to_index(i, out_shape, out_index)                                            |
                                                                                         |
            # Handle broadcasting between tensors to get input tensor indices            |
            broadcast_index(out_index, out_shape, a_shape, a_index)                      |
            broadcast_index(out_index, out_shape, b_shape, b_index)                      |
                                                                                         |
            # Convert indices to positions in storage                                    |
            a_pos = index_to_position(a_index, a_strides)     # First input position     |
            b_pos = index_to_position(b_index, b_strides)     # Second input position    |
            out_pos = index_to_position(out_index, out_strides) # Output position        |
                                                                                         |
            # Apply function and store result                                            |
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/stephen_shen/Deskto
p/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (260) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)  # Output tensor index
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/stephen_shen/Deskto
p/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (261) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)    # First input tensor
index
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/stephen_shen/Deskto
p/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (262) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)    # Second input tensor
index
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/stephen_shen/Desktop/ML_Engineering/workspace/mod3-
Stephen0512/minitorch/fast_ops.py (303)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/stephen_shen/Desktop/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (303)
-----------------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                                 |
        out: Storage,                                                                                            |
        out_shape: Shape,                                                                                        |
        out_strides: Strides,                                                                                    |
        a_storage: Storage,                                                                                      |
        a_shape: Shape,                                                                                          |
        a_strides: Strides,                                                                                      |
        reduce_dim: int,                                                                                         |
    ) -> None:                                                                                                   |
        # Calculate the size of the reduction dimension for the inner loop                                       |
        reduce_size = a_shape[reduce_dim]                                                                        |
                                                                                                                 |
        # Process each element in the output tensor in parallel                                                  |
        for i in prange(out.size):-------------------------------------------------------------------------------| #4
                                                                                                                 |
            # Create index buffers for input tensor index                                                        |
            index = np.empty(MAX_DIMS, np.int32)  # Tensor index for output first and then for input             |
                                                                                                                 |
            # Convert flat index to output index                                                                 |
            to_index(i, out_shape, index)                                                                        |
                                                                                                                 |
            # Convert output index to position in output tensor storage for final output update                  |
            out_pos = index_to_position(index, out_strides)                                                      |
                                                                                                                 |
            # Initialize reduction with first element of the reduction dimension in input tensor                 |
            index[reduce_dim] = 0                                                                                |
            in_pos = index_to_position(index, a_strides)  # Convert index to position in input tensor storage    |
                                                                                                                 |
            # Initialize accumulated value with the first element of the reduction dimension in input tensor     |
            accumulated_value = a_storage[in_pos]                                                                |
                                                                                                                 |
            # Inner reduction loop for each element in the reduction dimension (apart from the first one)        |
            for j in range(1, reduce_size):                                                                      |
                # Update index for next position in reduction dimension                                          |
                index[reduce_dim] = j                                                                            |
                in_pos = index_to_position(index, a_strides)                                                     |
                                                                                                                 |
                # Apply reduction function to accumulate result                                                  |
                accumulated_value = fn(accumulated_value, a_storage[in_pos])                                     |
                                                                                                                 |
            # Write final accumulated result to output tensor storage                                            |
            out[out_pos] = accumulated_value                                                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/stephen_shen/Deskto
p/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (319) is
hoisted out of the parallel loop labelled #4 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: index = np.empty(MAX_DIMS, np.int32)  # Tensor index for output
first and then for input
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/step
hen_shen/Desktop/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py
 (349)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/stephen_shen/Desktop/ML_Engineering/workspace/mod3-Stephen0512/minitorch/fast_ops.py (349)
---------------------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                           |
    out: Storage,                                                                                                                      |
    out_shape: Shape,                                                                                                                  |
    out_strides: Strides,                                                                                                              |
    a_storage: Storage,                                                                                                                |
    a_shape: Shape,                                                                                                                    |
    a_strides: Strides,                                                                                                                |
    b_storage: Storage,                                                                                                                |
    b_shape: Shape,                                                                                                                    |
    b_strides: Strides,                                                                                                                |
) -> None:                                                                                                                             |
    """NUMBA tensor matrix multiply function.                                                                                          |
                                                                                                                                       |
    Should work for any tensor shapes that broadcast as long as                                                                        |
                                                                                                                                       |
    ```                                                                                                                                |
    assert a_shape[-1] == b_shape[-2]                                                                                                  |
    ```                                                                                                                                |
                                                                                                                                       |
    Optimizations:                                                                                                                     |
                                                                                                                                       |
    * Outer loop in parallel                                                                                                           |
    * No index buffers or function calls                                                                                               |
    * Inner loop should have no global writes, 1 multiply.                                                                             |
                                                                                                                                       |
                                                                                                                                       |
    Args:                                                                                                                              |
    ----                                                                                                                               |
        out (Storage): storage for `out` tensor                                                                                        |
        out_shape (Shape): shape for `out` tensor                                                                                      |
        out_strides (Strides): strides for `out` tensor                                                                                |
        a_storage (Storage): storage for `a` tensor                                                                                    |
        a_shape (Shape): shape for `a` tensor                                                                                          |
        a_strides (Strides): strides for `a` tensor                                                                                    |
        b_storage (Storage): storage for `b` tensor                                                                                    |
        b_shape (Shape): shape for `b` tensor                                                                                          |
        b_strides (Strides): strides for `b` tensor                                                                                    |
                                                                                                                                       |
    Returns:                                                                                                                           |
    -------                                                                                                                            |
        None : Fills in `out`                                                                                                          |
                                                                                                                                       |
    """                                                                                                                                |
    # Calculate batch stride for tensor a and b                                                                                        |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                             |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                             |
                                                                                                                                       |
    # a = [[1, 2], [3, 4]] * b = [[5, 6], [7, 8]] = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]                                   |
    # Stride for moving to the next element in the row / column of tensor a                                                            |
    a_col_stride = a_strides[1]                                                                                                        |
    a_row_stride = a_strides[2]  # as mutiplication needs all the elements in the row for tensor a                                     |
                                                                                                                                       |
    # Stride for moving to the next element in the row / column of tensor b                                                            |
    b_col_stride = b_strides[1]  # as mutiplication needs all the elements in the column for tensor b                                  |
    b_row_stride = b_strides[2]                                                                                                        |
                                                                                                                                       |
    # The dimension for the result of each batch (must match: last dim of a, second-to-last of b)                                      |
    result_dim = b_shape[-2]                                                                                                           |
                                                                                                                                       |
    # Process each batch of the output tensor in parallel                                                                              |
    for batch_index in prange(out_shape[0]):-------------------------------------------------------------------------------------------| #5
                                                                                                                                       |
        # Process each element in the output tensor for the current batch                                                              |
        for row in range(out_shape[1]):                                                                                                |
            for col in range(out_shape[2]):                                                                                            |
                                                                                                                                       |
                # Calculate the first element in the row of tensor a for the current batch                                             |
                a_index = batch_index * a_batch_stride + row * a_col_stride                                                            |
                                                                                                                                       |
                # Calculate the first element in the column of tensor b for the current batch                                          |
                b_index = batch_index * b_batch_stride + col * b_row_stride                                                            |
                                                                                                                                       |
                # Calculate the position of the result in the output tensor for the current batch, row and column                      |
                out_index = batch_index * out_strides[0] + row * out_strides[1] + col * out_strides[2]                                 |
                                                                                                                                       |
                # Decalre a variable for the result of the products                                                                    |
                result = 0.0                                                                                                           |
                                                                                                                                       |
                # Inner product loop for the calculating the sum of the products of different parts of elements in tensor a and b      |
                for _ in range(result_dim):                                                                                            |
                    # Add the product of the elements pair in tensor a and b to the result                                             |
                    result += a_storage[a_index] * b_storage[b_index]                                                                  |
                                                                                                                                       |
                    # Update the indices for the next element in the row of tensor a and the next element in the column of tensor b    |
                    a_index += a_row_stride                                                                                            |
                    b_index += b_col_stride                                                                                            |
                                                                                                                                       |
                # Store the result in the output tensor storage                                                                        |
                out[out_index] = result                                                                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Task 3.5: Model Trainings

Two models were trained for each of the following datasets using CPU and GPU in Google Colab's GPU setup: Simple, Split, and XOR.

Several models were trained for each dataset with the same configuration, and the best results are included below for this part of the assignment.

### Task 3.5.1: Simple Dataset Model Training

#### Model Training Configuration

- Number of points: 50
- Size of hidden layer: 100
- Learning rate: 0.05
- Number of epochs: 500

#### CPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: 0.14159s

##### Output logs:

```console
Epoch  0  loss  9.400616814835383 correct 36
Epoch  10  loss  1.5833925528594972 correct 46
Epoch  20  loss  0.8033727623073732 correct 48
Epoch  30  loss  2.181663084881539 correct 48
Epoch  40  loss  1.3709837009904808 correct 49
Epoch  50  loss  1.121565921296268 correct 49
Epoch  60  loss  1.059402966082776 correct 49
Epoch  70  loss  0.7268074719132201 correct 50
Epoch  80  loss  0.2750735328662339 correct 49
Epoch  90  loss  0.7335653763236831 correct 49
Epoch  100  loss  0.31474121680844463 correct 50
Epoch  110  loss  0.36347241697870863 correct 49
Epoch  120  loss  0.37711988813392017 correct 50
Epoch  130  loss  0.5795507912669864 correct 49
Epoch  140  loss  1.2268163006902726 correct 49
Epoch  150  loss  0.38539331216869954 correct 50
Epoch  160  loss  1.2509134771348112 correct 49
Epoch  170  loss  0.009587360895539753 correct 48
Epoch  180  loss  1.1379073864103029 correct 50
Epoch  190  loss  0.6119981514223999 correct 49
Epoch  200  loss  0.08406249156874582 correct 50
Epoch  210  loss  0.1500414289935695 correct 50
Epoch  220  loss  0.39101834637440636 correct 49
Epoch  230  loss  0.2476367286877214 correct 49
Epoch  240  loss  0.6716398374696104 correct 50
Epoch  250  loss  0.0475882219387694 correct 49
Epoch  260  loss  0.21042151104858292 correct 49
Epoch  270  loss  0.8514013597085501 correct 49
Epoch  280  loss  0.1537635621325783 correct 49
Epoch  290  loss  0.06825685072210368 correct 49
Epoch  300  loss  0.3945544571330625 correct 49
Epoch  310  loss  0.7319965952163006 correct 50
Epoch  320  loss  0.886033008515392 correct 49
Epoch  330  loss  0.15898086975450346 correct 49
Epoch  340  loss  0.12269576427348257 correct 50
Epoch  350  loss  0.2187264557886995 correct 49
Epoch  360  loss  0.6543554038622658 correct 50
Epoch  370  loss  1.0453764422568632 correct 49
Epoch  380  loss  1.161796864333725 correct 50
Epoch  390  loss  0.2018117839235016 correct 49
Epoch  400  loss  0.6794099525178102 correct 50
Epoch  410  loss  0.028339995009706586 correct 50
Epoch  420  loss  0.3763310297379277 correct 49
Epoch  430  loss  0.058509050046136754 correct 50
Epoch  440  loss  0.9677776359477085 correct 50
Epoch  450  loss  0.8017641448620723 correct 50
Epoch  460  loss  0.7509906772826311 correct 50
Epoch  470  loss  0.3455250900638234 correct 50
Epoch  480  loss  0.058274616167516606 correct 50
Epoch  490  loss  0.17829775664636974 correct 50
Time per epoch: 0.14158626127243043 seconds
```

#### GPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: to be added...

##### Output logs:

```console
to be added...
```

### Task 3.5.2: Split Dataset Model Training

#### Model Training Configuration

- Number of points: 50
- Size of hidden layer: 100
- Learning rate: 0.05
- Number of epochs: 500

#### CPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: 0.13744s

##### Output logs:

```console
Epoch  0  loss  7.770038247455205 correct 26
Epoch  10  loss  7.118559635027194 correct 43
Epoch  20  loss  6.227517428823506 correct 42
Epoch  30  loss  4.022680024813758 correct 45
Epoch  40  loss  4.069979632193774 correct 47
Epoch  50  loss  3.1010444803368107 correct 46
Epoch  60  loss  3.4609087997606784 correct 50
Epoch  70  loss  2.5833040799800573 correct 48
Epoch  80  loss  2.6768432650877267 correct 49
Epoch  90  loss  1.2162799641531747 correct 48
Epoch  100  loss  1.4876963041597193 correct 50
Epoch  110  loss  0.7200835577232305 correct 48
Epoch  120  loss  1.20013992452978 correct 50
Epoch  130  loss  1.2409467532346603 correct 50
Epoch  140  loss  0.8022590983976798 correct 50
Epoch  150  loss  1.4416032269497776 correct 50
Epoch  160  loss  0.5152863094243957 correct 50
Epoch  170  loss  0.6144135114870541 correct 50
Epoch  180  loss  0.3257852228667009 correct 50
Epoch  190  loss  0.4300947341095828 correct 50
Epoch  200  loss  0.44895441615965515 correct 50
Epoch  210  loss  1.0111194835995048 correct 50
Epoch  220  loss  0.7655668135303636 correct 50
Epoch  230  loss  0.4433794236516998 correct 50
Epoch  240  loss  0.2595219494907319 correct 50
Epoch  250  loss  0.40250267803764667 correct 50
Epoch  260  loss  0.25675448116754035 correct 50
Epoch  270  loss  0.36268259945732206 correct 50
Epoch  280  loss  0.48460940375768585 correct 50
Epoch  290  loss  0.24752631708406606 correct 50
Epoch  300  loss  0.059803992162497745 correct 50
Epoch  310  loss  0.5023992721954529 correct 50
Epoch  320  loss  0.3575245922008051 correct 50
Epoch  330  loss  0.03141163991174711 correct 50
Epoch  340  loss  0.3971080988484158 correct 50
Epoch  350  loss  0.06221017031821714 correct 50
Epoch  360  loss  0.051207519051606046 correct 50
Epoch  370  loss  0.5083717378943033 correct 50
Epoch  380  loss  0.37919302006559275 correct 50
Epoch  390  loss  0.3690727697385319 correct 50
Epoch  400  loss  0.15762415385809642 correct 50
Epoch  410  loss  0.43365781438005035 correct 50
Epoch  420  loss  0.3024658612257021 correct 50
Epoch  430  loss  0.340510497140175 correct 50
Epoch  440  loss  0.25334954970799545 correct 50
Epoch  450  loss  0.27156641079875554 correct 50
Epoch  460  loss  0.17466032764322292 correct 50
Epoch  470  loss  0.27835552886038606 correct 50
Epoch  480  loss  0.23349397165407915 correct 50
Epoch  490  loss  0.2424684042379801 correct 50
Time per epoch: 0.13743920278549193 seconds
```

#### GPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: to be added...

##### Output logs:

```console
to be added...
```

### Task 3.5.3: XOR Dataset Model Training

#### Model Training Configuration

- Number of points: 50
- Size of hidden layer: 100
- Learning rate: 0.05
- Number of epochs: 500

#### CPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: 0.13798s

##### Output logs:

```console
Epoch  0  loss  7.571220737043088 correct 28
Epoch  10  loss  4.723169194424073 correct 36
Epoch  20  loss  4.44512142312101 correct 36
Epoch  30  loss  4.772882445185869 correct 44
Epoch  40  loss  3.826549920228981 correct 45
Epoch  50  loss  2.156360599444318 correct 41
Epoch  60  loss  1.791871504345767 correct 39
Epoch  70  loss  4.442584806487662 correct 44
Epoch  80  loss  2.3646817313155446 correct 47
Epoch  90  loss  2.077618621018851 correct 50
Epoch  100  loss  1.853705953202932 correct 47
Epoch  110  loss  2.245249815189637 correct 47
Epoch  120  loss  1.7428612972518818 correct 50
Epoch  130  loss  2.4457602415877475 correct 49
Epoch  140  loss  1.104478404355583 correct 46
Epoch  150  loss  2.376392034561654 correct 48
Epoch  160  loss  1.9488866670453593 correct 50
Epoch  170  loss  1.659120362855388 correct 49
Epoch  180  loss  2.1691012142903334 correct 47
Epoch  190  loss  1.207543091621801 correct 50
Epoch  200  loss  0.41696379790691035 correct 48
Epoch  210  loss  1.088984619793311 correct 49
Epoch  220  loss  2.0401871463829084 correct 49
Epoch  230  loss  1.5711295534552951 correct 50
Epoch  240  loss  0.35553248415397853 correct 50
Epoch  250  loss  1.2531707931896623 correct 50
Epoch  260  loss  0.9130762200682444 correct 49
Epoch  270  loss  1.0825340895587152 correct 50
Epoch  280  loss  0.7048083893208021 correct 49
Epoch  290  loss  0.9350083765418906 correct 50
Epoch  300  loss  1.1869513278449944 correct 50
Epoch  310  loss  0.8351736543830424 correct 50
Epoch  320  loss  0.5618033154666471 correct 50
Epoch  330  loss  1.1114971227225872 correct 49
Epoch  340  loss  0.813889355416295 correct 50
Epoch  350  loss  1.6420528038217326 correct 48
Epoch  360  loss  0.4561708472632626 correct 50
Epoch  370  loss  0.25311597611764947 correct 50
Epoch  380  loss  1.3450182513433175 correct 48
Epoch  390  loss  0.5489844271992189 correct 49
Epoch  400  loss  0.5061228282943697 correct 50
Epoch  410  loss  0.34348146376174726 correct 50
Epoch  420  loss  0.44233355180932654 correct 50
Epoch  430  loss  0.6523500094441141 correct 50
Epoch  440  loss  0.9589943811486318 correct 50
Epoch  450  loss  0.5844034149615487 correct 50
Epoch  460  loss  0.3604295822765346 correct 50
Epoch  470  loss  0.048966122173029446 correct 50
Epoch  480  loss  0.7638276814764614 correct 50
Epoch  490  loss  0.7303678694853736 correct 50
Time per epoch: 0.13798286485671998 seconds
```

#### GPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: to be added...

##### Output logs:

```console
to be added...
```

### Task 3.5.4: Model Training with Large Size of Hidden Layer

For this part of the assignment, two models for XOR dataset were trained with a larger size of hidden layer (200 layers) using CPU and GPU in Google Colab's GPU setup.

#### Model Training Configuration

- Number of points: 50
- Size of hidden layer: 200
- Learning rate: 0.05
- Number of epochs: 500

#### CPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: 0.25283s

##### Output logs:

```console
Epoch  0  loss  12.886690643951422 correct 21
Epoch  10  loss  3.6290676031694002 correct 42
Epoch  20  loss  1.9084421170634782 correct 47
Epoch  30  loss  2.1035888732955637 correct 47
Epoch  40  loss  1.8385565315757504 correct 48
Epoch  50  loss  2.1648885810944782 correct 48
Epoch  60  loss  0.8223768654497151 correct 46
Epoch  70  loss  1.5281144649068035 correct 48
Epoch  80  loss  1.6900116361872661 correct 46
Epoch  90  loss  1.0324218321819842 correct 48
Epoch  100  loss  1.3039723565465577 correct 48
Epoch  110  loss  1.3518570042588527 correct 49
Epoch  120  loss  2.1921494225385953 correct 49
Epoch  130  loss  0.9782356335324711 correct 49
Epoch  140  loss  0.28201226293445686 correct 49
Epoch  150  loss  1.3195234269462215 correct 50
Epoch  160  loss  0.4345236573339819 correct 50
Epoch  170  loss  0.5492434110890689 correct 49
Epoch  180  loss  0.8444784953137129 correct 50
Epoch  190  loss  0.1633870045647372 correct 50
Epoch  200  loss  0.41774907183782256 correct 50
Epoch  210  loss  0.6309014649114412 correct 49
Epoch  220  loss  0.7606799704663652 correct 49
Epoch  230  loss  0.7011422654409895 correct 50
Epoch  240  loss  0.7617527969205331 correct 50
Epoch  250  loss  0.5479392051121191 correct 50
Epoch  260  loss  0.5206838809023014 correct 50
Epoch  270  loss  0.3619415729828762 correct 50
Epoch  280  loss  0.5923086697168467 correct 50
Epoch  290  loss  0.5039296453271545 correct 50
Epoch  300  loss  0.46997081964281073 correct 50
Epoch  310  loss  0.5320379972023466 correct 50
Epoch  320  loss  0.3151483425059503 correct 50
Epoch  330  loss  0.2615861584502418 correct 50
Epoch  340  loss  0.07516002094823651 correct 50
Epoch  350  loss  0.4803186967940823 correct 50
Epoch  360  loss  0.49813273680879455 correct 50
Epoch  370  loss  0.5446609234099241 correct 50
Epoch  380  loss  0.8349737142939927 correct 50
Epoch  390  loss  0.28785533237384264 correct 50
Epoch  400  loss  0.3386502715121997 correct 50
Epoch  410  loss  0.052434576199085495 correct 50
Epoch  420  loss  0.5622776703350532 correct 50
Epoch  430  loss  0.19334876538761694 correct 50
Epoch  440  loss  0.5398206311219165 correct 50
Epoch  450  loss  0.09990801374311194 correct 50
Epoch  460  loss  0.08317071165451159 correct 50
Epoch  470  loss  0.6509413684372071 correct 50
Epoch  480  loss  0.2102027736907837 correct 50
Epoch  490  loss  0.24454864692092776 correct 50
Time per epoch: 0.2528335914611816 seconds
```

#### GPU Model Training Result

##### Command used:

```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05
```

##### Resulted Time Per Epoch

- Time per epoch: to be added...

##### Output logs:

```console
to be added...
```

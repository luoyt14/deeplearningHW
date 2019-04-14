| experiments  | test_acc | test_loss |
| :----------: | -------- | --------- |
| experiment 1 | 0.983    | 0.052     |
| experiment 2 | 0.987    | 0.048     |
|     MLP      | 0.983    | 0.021     |

## experiment 1:

$$
Input (1\times 28 \times 28) \\
Conv1 (4\times 3 \times 3, padding=1)+Relu \\
AvgPool(2\times 2, padding=0) (Output: 4\times 14 \times 14)\\
Conv1 (4\times 3 \times 3, padding=1)+Relu \\
AvgPool(2\times 2, padding=0) (Output: 4\times 7 \times 7)\\
Flatten(196) \\
Linear(196\times 10)
$$

## experiment 2

$$
Input (1\times 28 \times 28) \\
Conv1 (8\times 3 \times 3, padding=1)+Relu \\
AvgPool(2\times 2, padding=0) (Output: 8\times 14 \times 14)\\
Conv1 (16\times 3 \times 3, padding=1)+Relu \\
AvgPool(2\times 2, padding=0) (Output: 16\times 7 \times 7)\\
Flatten(784) \\
Linear(784\times 256)+Relu \\
Linear(156\times 10)
$$


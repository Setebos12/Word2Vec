# CBOW (Continuous Bag of Words) – Batch Version



## 1. Softmax and Loss (Batch Mode)
In the batch version, operations are performed on matrices where the first dimension corresponds to the batch size $B$.

### `softmax_probs(logits)`
Computes the softmax distribution for a batch of logit vectors. To ensure numerical stability (preventing float overflow), we subtract the maximum logit value from each row.

**Formula:**
$$\text{softmax}(z_{b,i}) = \frac{\exp(z_{b,i} - \max(z_b))}{\sum_j \exp(z_{b,j} - \max(z_b))}$$

* **$b$**: Index of the example in the batch.
* **$i$**: Index of the word in the vocabulary.

---

## 2. Batch Creation
### `create_batches(words_indexes, window_size, batch_size)`
Divides the corpus into randomized batches. For each target word, a context is generated using a sliding window.

**Logic:**
1.  **Shuffle**: Randomize word indices (optional but recommended for convergence).
2.  **Context Extraction**: For each target word $w_t$, extract:
    $$C_t = \{w_{t-s}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+s}\}$$
3.  **Grouping**: Aggregate these pairs into matrices of size $B$.

---

## 3. Forward Pass (Batch)
### `forward_pass_batch(V, V_prime, context_batch, target_batch)`
Performs the projection and output calculation for $B$ examples simultaneously.

**Vectorized Steps:**
1.  **Hidden State Batch ($h_{batch}$)**: The average of context embeddings for each row in the batch.
    $$h_b = \frac{1}{|C_b|} \sum_{w \in C_b} V_w$$
2.  **Logits Batch ($z_{batch}$)**:
    $$z = h \cdot V'^T$$
3.  **Average Cross-Entropy Loss**:
    $$L_{avg} = \frac{1}{B} \sum_{b=1}^{B} -\log(y_{b, \text{target}})$$

---

## 4. Backward Pass (Batch) – Detailed Derivatives
The `backward_pass_batch` function computes gradients for an entire batch using backpropagation via the chain rule.

### 1. Gradient w.r.t. Logits (Output Error)
For a single example, the derivative of the Cross-Entropy loss w.r.t. logits $z_i$ ($\delta$) simplifies to:
$$\frac{\partial L}{\partial z_i} = y_i^{(pred)} - y_i^{(true)}$$

**Batch Version:**
We combine error vectors into a matrix of shape $(B \times \text{vocab\_size})$:
$$\delta_{batch} = Y_{probs} - Y_{one\_hot}$$

---

### 2. Gradient w.r.t. Output Weight Matrix ($V'$)
Using the chain rule, the gradient with respect to the output weights is the outer product of the error and the hidden state.
**Batch Result:**
$$\nabla V' = \delta^T \cdot h$$

* **Dimensions**: $(\text{vocab\_size} \times B) \cdot (B \times \text{embedding\_dim}) = (\text{vocab\_size} \times \text{embedding\_dim})$
* The transposition of $\delta^T$ effectively sums the contributions from all examples in the batch.

---

### 3. Gradient w.r.t. Hidden Layer ($h$)
To propagate the error back to the input embeddings, we calculate how the hidden layer affects the loss:
$$\nabla h = \delta_{batch} \cdot V'$$

* **Dimensions**: $(B \times \text{vocab\_size}) \cdot (\text{vocab\_size} \times \text{embedding\_dim}) = (B \times \text{embedding\_dim})$

---

### 4. Gradient w.r.t. Input Embeddings ($V$)
Since $h_b$ is the average of context word embeddings, the derivative of $h_b$ w.r.t. a single input embedding $V_{w_c}$ is $\frac{1}{|C_b|}$.

**Weight Update (Gradient Descent):**
For each context word $w_c$ in example $b$ of the batch:
$$V_{w_c} \gets V_{w_c} - \eta \cdot \left( \frac{1}{|C_b|} \nabla h_b \right)$$

---

## 5. Weights Update
### `update_weights_batch(...)`
Standard gradient descent updates:
* $V' \gets V' - \eta \nabla V'$
* $V \gets V - \eta \nabla V$
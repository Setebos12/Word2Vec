# CBOW no batch

## 1. Softmax and Log-Softmax

These functions transform raw scores (logits) into a probability distribution.

### `softmax_probs(x)`
Calculates the probabilities for a vector of logits $x$. A numerically stable variant is used to prevent overflow.
* **Reference:** [Softmax Function (Wikipedia)](https://en.wikipedia.org/wiki/Softmax_function)

**Formula:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Stable Implementation:**
$$\text{softmax}(x) = \frac{\exp(x - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

### `log_softmax(x)`
Calculates the logarithm of the softmax function, which is essential for stable numerical computation during backpropagation.

**Formula:**
$$\log \text{softmax}(x_i) = x_i - \log \sum_j e^{x_j}$$

---

### `get_corpus(data, target_index, size)`
 **[Continuous bag-of-words (CBOW)](https://en.wikipedia.org/wiki/Word2vec#Continuous_bag-of-words_(CBOW))**.

**Formula:**
For a target word $w_t$ and window size $s$:
$$C_t = \{ w_{t-s}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+s} \}$$

---

## 3. Forward Pass

### `fastforward(V, V_prime, context_indexes)`
Computes the flow of data from input embeddings to output probabilities.

**Computational Steps:**
1. **Hidden Layer Embedding** (Average of context vectors):
   $$h = \frac{1}{|C_t|} \sum_{w_c \in C_t} V_{w_c}$$
2. **Logits for Softmax**:
   $$z = V' \cdot h$$
3. **Probabilities**:
   $$y = \text{softmax}(z)$$

---

## 4. Loss Function

### `cross_entropy_loss(logits, target_index)`
Measures the prediction error of the model for the target word $t$.
* **Reference:** [Cross-entropy (Wikipedia)](https://en.wikipedia.org/wiki/Cross-entropy)

**Formula:**
$$L = - \log \text{softmax}(z_{t})$$

---

## 5. Backward Pass and Gradients

Calculates the gradients required to update the model weights using the chain rule.



**Gradient Formulas:**
1. **Gradient w.r.t. Logits ($z$):**
   The derivative of the Cross-entropy loss with respect to the Softmax input simplifies to:
   $$\frac{\partial L}{\partial z_i} = y_i - \mathbf{1}_{[i = t]}$$
2. **Gradient w.r.t. Output Matrix ($V'$):**
   $$\frac{\partial L}{\partial V'} = \left(\frac{\partial L}{\partial z}\right) \otimes h$$
3. **Gradient w.r.t. Hidden Layer ($h$):**
   $$\frac{\partial L}{\partial h} = V'^T \cdot \frac{\partial L}{\partial z}$$
4. **Gradient w.r.t. Input Embeddings ($V_c$):**
   Since $h$ is a simple average, the gradient is distributed equally among context words:
   $$\frac{\partial L}{\partial V_c} = \frac{1}{|C_t|} \frac{\partial L}{\partial h}$$

---

## 6. Weight Update

### `update_weights(V, V_prime, context_indices, grad_V, grad_V_prime, lr)`
Updates the model parameters using the standard Gradient Descent algorithm.

**Update Formulas:**
* $$V' \gets V' - \eta \frac{\partial L}{\partial V'}$$
* $$V_c \gets V_c - \eta \frac{\partial L}{\partial V_c}$$
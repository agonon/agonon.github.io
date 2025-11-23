

I like working on projects that involve:
- **◈** respecting the **structure and geometry** of problems (e.g., invariances in theory/algorithms; structured sparsity for efficiency)  
- **△** investigating **minimal setups** that still capture key large-scale effects  
- **⇆** bringing **theory closer to practice**, with the aim of better understanding, informing, or improving the latter

Below are small introductory reads to a few selected areas I’ve worked on.

*Legend:* **◈** structure & geometry · **△** minimal setups · **⇆** theory ↔ practice  
*(Ongoing projects in the △ direction; more soon.)*


<div class="roles cta-row" style="margin: 8px 0 16px;">
  <a class="btn ghost" id="expand-all" href="#">Expand all</a>
  <a class="btn ghost" id="collapse-all" href="#" style="margin-left:8px;">Collapse all</a>
</div>

---

<!-- Section 1 -->
<details class="interest">
  <summary>
    <span class="interest-title">Generalization guarantees that respect parameter symmetries ◈ ⇆</span>
    <span class="readtime" aria-label="Estimated reading time"></span>
  </summary>
  <div class="interest-content" markdown="1">
**Related papers:**
<div class="minibib">
  <span id="S1R1" class="refnum">[1]</span>
  Rouchouse*, Gonon*, Gribonval, Guedj, <em>Non-Vacuous Generalization Bounds: Can Rescaling Invariances Help?</em>, arXiv 2025
  · <a href="https://arxiv.org/pdf/2509.26149" target="_blank" rel="noopener">PDF</a>

  <br>

  <span id="S1R2" class="refnum">[2]</span>
  Gonon, Brisebarre, Riccietti, Gribonval, <em>A Path-Norm Toolkit for Modern Networks</em>, ICLR 2024 (Spotlight)
  · <a href="https://arxiv.org/pdf/2310.01225" target="_blank" rel="noopener">PDF</a>
</div>


Generalization refers to the statistical question of whether a neural network's performance on unseen data matches its performance on the training set. Understanding the factors that influence generalization is essential for interpreting neural network behavior and using these insights to design more effective models.

The generalization gap of a model is statistical quantity that is defined to measure how the model's performance on unseen data deviates from its performance on the training set. Often it’s small, sometimes it isn’t (e.g., <a href="https://en.wikipedia.org/wiki/Grokking_(machine_learning)" target="_blank" rel="noopener">grokking</a>). The question is: which properties seen in practice are enough to make this gap small? Knowing this matters to **explain** results and to design more **effective** models by ensuring those properties appear more reliably.

Two routes I’ve studied to bound the generalization gap are **Rademacher** and **PAC-Bayes** bounds. A recurring issue is **parameter symmetries**: the parameterization of neural networks is often redundant, i.e., different parameter vectors can represent the **same function**, hence the same generalization behavior, yet many bounds still treat these parameterizations as different. Among these symmetries are the so-called "rescaling-symmetries", which are particularly significant because: (1) they create continuous surfaces of local minima, unlike discrete permutation symmetries, and (2) they modify the size of the weights, unlike (again) permutation symmetries. 

A simple place to see it is a single ReLU neuron $f(x)=v\max(0,ux)$. Scaling $u$ and $v$ by reciprocal factors leaves $f$ unchanged (i.e., $u\to \lambda u$, $v\to \lambda^{-1}v$ with $\lambda>0$), however, it changes the size of the weights and, in particular, it can affect parameter-space measures (e.g., the squared $\ell^2$ norm $u^2 + v^2 $) which might not be invariant.  Theoretical guarantees should often account for these symmetries to avoid being arbitrarily pessimistic when weights are rescaled. For instance, a generalization bound that scales with the squared $\ell^2$ norm of the weights can be made arbitrarily large by rescaling $u$ and $v$ in opposite directions, even though the function and its generalization behavior remain unchanged. 

A first possible remedy is to take the infimum of the complexity measure over all rescalings (e.g., the infimum of the rescaled squared $\ell^2$-norm $(\lambda u)^2 + (v/\lambda)^2$ over $\lambda>0$ is simply the $\ell^1$-norm $\|u\| + \|v\|$), but in general, computing the infimum over all possible rescalings is often not trivial for the type of parameter-space complexity measures actually showing up in bounds.

Another approach is to work with invariant representations of the weights. For the simple example above, the product $uv$ serves as a basic invariant: $uv = (\lambda u)(v/\lambda)$ for any $\lambda>0$. Any complexity measure based on $uv$ will be invariant to the rescaling symmetry. The **path-lifting** generalizes this idea to more complex networks, regrouping similar invariants made of products of weights into a single vector. This vector is closely tied to the function implemented by the network, which is great for theoretical analysis. Before my work, the path-lifting was defined for multi-layer perceptrons. While this is already very useful, it could not handle skip connections, max-pooling, and even sometimes biases, or multi-dimensional outputs. In <a href="#S1R2" class="refcite">[2]</a>, I extended path-lifting and its related tools to modern networks such as ResNets, VGGs, U-Nets, ReLU MobileNets, AlexNet, AlphaGo, and more.

A natural question that follows is: can we derive guarantees **in terms of an invariant representation** of the weights such as the path-lifting? I have studied this question in two different frameworks for bounding the generalization gap: Rademacher and PAC-Bayes.

A quick refresher: consider some i.i.d. data $(x_i,y_i)$ (e.g., $x_i$ is an image and $y_i$ is its label, like "cat" or "dog"). Generalization bounds for a predictor $f_w$ with parameters $w$ typically aim to control the **train–test gap** (or generalization gap) defined as the difference between the test error (expected loss) $\mathbb{E}_{(x,y)}[\ell(f_w(x),y)]$ on the whole (unknown) distribution of the data $(x,y)$ and the empirical average (train error) $\frac{1}{n}\sum_i \ell(f_w(x_i),y_i)$. 
If the weights $w$ of the model were chosen independently of the available data $(x_i,y_i)$, then classical concentration would suffice to bound this difference. However, in practice $w$ is chosen to make sure that $f_w$ is not too bad to predict $y_i\approx f_w(x_i)$ on the training data, so $w$ is **data-dependent** (we say $x$ has been trained on $(x_i,y_i)$). This makes the terms in the sum dependent too, and classical concentration no longer applies. This is where statistical learning theory comes in: it provides tools to control this dependence. Two frameworks for this are Rademacher and PAC-Bayes bounds.

- **Rademacher approach <a href="#S1R2" class="refcite">[2]</a>.** One workaround is to bound the **worst-case** train–test gap over all possible weight vectors in a set $W$, which contains our given $w$ of interest. Since $W$ is fixed before seeing the data, one can use concentration to control the gap for any given element, and Rademacher complexity measures how rich $W$ is to control the worst-case concentration uniformly over $W$. The caveat of this approach is that the way it restores data-independence (in order to fall back on concentration results) is too crude to stay informative: data dependence becomes very loose in the final bound (e.g., reduced to coarse quantities like $\max_i \|x_i\|$), so the bound is many orders of magnitude too large in some of the training regimes that are relevant today (e.g., hundreds of millions of parameters, far fewer data). Still, these bounds are a historical starting point, so I used them as a **first testbed** to refine invariance-aware guarantees: I brought **path-lifting** tools to express **Lipschitz properties in $w$ and in $x$** in an invariant way and ported them into Rademacher-style bounds for modern DAG-like ReLU nets (biases, pooling, skips). This unified and overall improved earlier lines (pure MLPs vs. DAGs, exponential-in-depth artifacts, etc., see Table 1 in <a href="#S1R2" class="refcite">[2]</a>) and extended the type of models for which this invariant machinery applies, bringing it closer to today’s practice by handling, e.g., ResNets. However, Rademacher bounds remain by nature worst-case in too many ways, and hence numerically vacuous in interesting regimes, which is why the next natural step was to turn to routes that keep more data dependence, like PAC-Bayes.

- **PAC-Bayes approach <a href="#S1R1" class="refcite">[1]</a>.** In simple terms, PAC-Bayes lets us replace worst-case data-independent bounds by in average, data-dependent, ones. More precisely, PAC-Bayes refines the Rademacher approach from worst-case to expected-case as follows: instead of ultimately measuring the **worst-case** concentration for parameters $w_{\textrm{data indep}}$ in a set $W$, it instead measures the **expected** concentration for $w_{\textrm{data indep}}\in W$ drawn according to a **distribution** $P$. The distribution $P$ is called a "prior" distribution and is fixed before seeing the data, therefore any parameters $w_{\textrm{data indep}}$ drawn according to $P$ do not depend on the available data $(x_i,y_i)$ and classical concentration theory applicable to any $w_{\textrm{data indep}}\sim P$. The very nice feature of PAC-Bayes is that, despite relying ultimately on a data-independent argument for $w_{\textrm{data indep}}\sim P$, it is still able to retain data dependence: given a data-dependent "posterior" distribution $Q$ over $W$ (e.g., a Gaussian around some weights $w$ obtained from fitting the data $(x_i,y_i)$, so any $w_{\textrm{data dep}}\sim Q$ depends on the available data $(x_i,y_i)$), PAC-Bayes bounds guarantee that the **expected** train–test gap under $Q$ is at most the expected gap under $P$ (the latter is easy to control by concentration) plus a distance between $Q$ and $P$ (often a KL-divergence). This shows that if there is a data-independent distribution $P$ that is not too far from the data-dependent $Q$, then the expected gap under $Q$ is small. In practice, this approach has produced informative bounds even for overparameterized models (more parameters than data, see our experiments on CIFAR-10 in <a href="#S1R1" class="refcite">[1]</a> for a simple example with ~5M parameters vs ~50K data), a regime that is usually challenging for classical statistical learning theory. 

However, the symmetry issue reappears in the PAC-Bayes **distance term** between $P$ and $Q$: the comparison of these two distributions shall be independent if one decides to rescale the weights in $Q$ (e.g., if $Q$ is a Gaussian with parameters defined in terms of $w$, then rescaling $w$ changes the mean and variance of $Q$, hence its distance to $P$). Again, this is problematic since two distributions that lead to the **same functions** can yield very different KLs. Our proposed remedy in <a href="#S1R1" class="refcite">[1]</a> is to **build invariance into the construction**: either optimize the bound over rescalings, or push $P$ and $Q$ through a **lift** that collapses rescaling-equivalent parameters before measuring divergence (e.g., the path-lifting discussed above for Rademacher). We show in practice that this can shrink the complexity term and turn vacuous bounds into non-vacuous ones <a href="#S1R1" class="refcite">[1]</a>.

**Takeaway.** If two parameterizations compute the same function, bounds should treat them the same.
  </div>
</details>

---

<!-- Section 2 -->
<details class="interest">
  <summary>
    <span class="interest-title">Symmetry-aware Bayesian optimization ◈ ⇆</span>
    <span class="readtime" aria-label="Estimated reading time"></span>
  </summary>
  <div class="interest-content" markdown="1">
**Related paper:**
<div class="minibib">
  <span id="S2R1" class="refnum">[1]</span>
  Bardou, Gonon, Ahadinia, Thiran, <em>Symmetry-Aware Bayesian Optimization via Max Kernels</em>, arXiv 2025
  · <a href="https://arxiv.org/pdf/2509.25051" target="_blank" rel="noopener">PDF</a>
</div>

Bayesian optimization (BO) aims at maximizing expensive-to-evaluate black-box objectives $f$ without accessing gradients (0-th order optimization) and minimizing the number of evaluations of $f$. There are plenty of black-box expensive-to-evaluate functions $f$ to be found in physics (e.g., take as input big-bang hyperparameters and simulate the early universe). In order to maximize $f$ with as few evaluations as possible, BO maintains a probabilistic model of $f$ (i.e., a distribution over functions, whose mean is basically a best guess of $f$ and whose variance quantifies uncertainty). This model is updated after each evaluation of $f$ and used to select the next point to evaluate, balancing exploration (evaluating where uncertainty is high) and exploitation (evaluating where the model predicts high values of $f$). The probabilistic model is defined via a **kernel** $k$ that encodes prior knowledge on how points in the input space are similar in terms of values of $f$: points with large $k(x,x')$ should have similar $f(x)$ and $f(x')$. BO intuitively uses $k$ to determine the new points on which $f$ should be evaluated (either very dissimilar points to explore, or points similar to known good ones to exploit). Therefore, the choice of $k$ is crucial to BO’s performance. 

When $f$ has **known input symmetries** (e.g., rotations of an image  $x$ does not change its label $f(x)$), the kernel should reflect that and rank as similar all inputs equivalent under these symmetries. A common trick to enforce that is to start with a non-invariant kernel (e.g., for images, one based on $\ell^2$-distance) and to **average** the values of this non-invariant kernel over all transformations in the symmetry group: $k(x,x') = \mathbb{E}_{g,g' \sim G}[k_0(g \cdot x, g' \cdot x')]$, where $G$ is the symmetry group and $k_0$ is the non-invariant kernel. However, for two images of cats, there is usually at most **one** transformation that closely aligns them so that, say, Euclidean distance reflects similarity. Averaging that single good alignment with many poor ones **dilutes** the signal. 

Instead, in <a href="#S2R1" class="refcite">[1]</a>, we reconsider an old idea: compare two points by their **best** alignment, not the average. The max kernel $k(x,x') = \max_{g,g' \in G}[k_0(g \cdot x, g' \cdot x')]$ isn’t PSD in general, which explains why it isn’t the default in current BO literature. To address this, we make it PSD **on the fly**: compute the Gram matrix on the current dataset, **clip negative eigenvalues** to project onto the PSD cone, and extend out-of-sample with **Nyström**. We report in <a href="#S2R1" class="refcite">[1]</a> that this simple adaptation leads to strong improvements on standard BO benchmarks with symmetries and a real Wi-Fi configuration problem, and it doesn’t worsen asymptotic complexity compared to standard BO (BO already inverts a Gram matrix at some point; our PSD projection costs the same order via an SVD). We also observe that current theory doesn’t yet explain the empirical gains; we discuss why that might be and how practice could suggest new theoretical paths.

**Takeaway.** Keep the invariance, but don’t average away the signal.
</div>
</details>

---


<!-- Section 3 -->
<details class="interest">
<summary>
<span class="interest-title">Invariant Lipschitz bounds and pruning ◈ ⇆</span>
<span class="readtime" aria-label="Estimated reading time"></span>
</summary>
  <div class="interest-content" markdown="1">
**Related papers:**
<div class="minibib">
  <span id="S3R1" class="refnum">[1]</span>
  Gonon, Brisebarre, Riccietti, Gribonval, <em>A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations</em>, ICML 2025
  · <a href="https://arxiv.org/pdf/2405.15006" target="_blank" rel="noopener">PDF</a>
  <br>
  <span id="S3R2" class="refnum">[2]</span>
  Gonon, Brisebarre, Riccietti, Gribonval, <em>A Path-Norm Toolkit for Modern Networks</em>, ICLR 2024 (Spotlight)
  · <a href="https://arxiv.org/pdf/2310.01225" target="_blank" rel="noopener">PDF</a>
</div>

I studied two types of Lipschitzness in neural networks: Lipschitzness with respect to the inputs and Lipschitzness with respect to the weights. **Lipschitzness with respect to the inputs** is critical for robustness to adversarial attacks. In <a href="#S3R2" class="refcite">[2]</a>, I extended an existing Lipschitz bound (previously defined for ReLU multi-layer perceptrons with scalar outputs) to modern architectures, including ResNets. This bound is based on the path-norm and offers several advantages: it improves over naive Lipschitz bounds (products of layers' norms), it is easy to compute (one forward pass), and it is invariant to rescaling-symmetries. This has been used in the same paper to derive Rademacher generalization bounds (discussed above on this page).  

**Lipschitzness with respect to the weights** is also important for various properties, including generalization or robustness to quantization. In <a href="/papers/#gonon-2025-pathmetric-lip" class="papercite">[ICML 2025](Lipschitz in weight space, pruning)</a>, I derived a novel Lipschitz bound in the weights based on the path-lifting vector. This bound is invariant to rescaling-symmetries, computable (two forward passes), and applies to modern architectures (ResNets, biases, pooling). I have used this bound to design a pruning rule as described next.

**Pruning** involves setting some neural network weights to zero, reducing the number of parameters used in the computations (zeros can be skipped when doing multiplications/additions) and potentially improving computational efficiency. A common approach is magnitude pruning, where weights with the smallest absolute values are pruned (set to zero), with the belief that since these weights are small, 
they contribute less to the network's output and thus can be removed with minimal impact on performance. However, neural networks often exhibit redundancies in their parameterization, meaning that different sets of weights can represent the same function. An example of this is the rescaling-symmetry discussed above, which is critical for pruning since it can change weight magnitudes without changing the function. A simple example is a ReLU network $f(x) = (v_1+v_2) \max(0,ux)$ with three scalar parameters $(u,v_1,v_2)$. For any $\lambda>0$, the rescaled weights $(\lambda u, v_1/\lambda, v_2/\lambda)$ implement the *same* function. And choosing $\lambda$ large or small can either make $u$ small (and hence, pruned if we do magnitude pruning) or large (and hence, kept). In this case, this completely changes the result of magnitude pruning: if we prune one weight among the three, and it happens to be $u$, then the pruned network is constant zero, while if we prune $v_1$ or $v_2$, the pruned network has still the same expressivity as the original one. In <a href="#S3R1" class="refcite">[1]</a>, I show at a larger scale (ResNets, ImageNet) that, indeed, magnitude pruning can yield drastically different results depending on the rescaling-equivalent set of weights it is applied to.

A natural remedy would be to take a pruning decision based on an invariant measure of weight importance. In <a href="#S3R1" class="refcite">[1]</a>, I propose to use the above-mentioned Lipschitz bound in weight space as such an invariant measure of importance. Indeed, this bound precisely quantifies how much the output of the network can change when a given weight is perturbed, only via *rescaling-invariant* quantities. Therefore, it is a natural candidate to measure the importance of weights in a way that respects the rescaling-symmetries of the model. I show in <a href="#S3R1" class="refcite">[1]</a> that pruning based on this invariant measure leads to the same performance than magnitude pruning in a proof-of-concept experiment with a ResNet on ImageNet, while it is robust to rescalings (unlike magnitude pruning). 
This pruning rule is based on the path-lifting, an invariant representation of neural networks that I extended to DAG-ReLU architectures (with pooling and skip connections) in <a href="#S3R2" class="refcite">[2]</a>.

**Takeaway.** Use invariant measures to make invariant decisions.
  </div>
</details>

---


<!-- Section 4 -->
<details class="interest">
  <summary>
    <span class="interest-title">Structure for speed: Kronecker-sparse inference on GPU ◈ ⇆</span>
    <span class="readtime" aria-label="Estimated reading time"></span>
  </summary>
  <div class="interest-content" markdown="1">
**Related paper:**
<div class="minibib">
  <span id="S4R1" class="refnum">[1]</span>
  Gonon*, Zheng*, Carrivain*, Le, <em>Fast Inference with Kronecker-Sparse Matrices</em>, ICML 2025
  · <a href="https://arxiv.org/pdf/2405.15013" target="_blank" rel="noopener">PDF</a>
  · <a href="https://github.com/PascalCarrivain/ksmm" target="_blank" rel="noopener">Code</a>
</div>

Matrix multiplication is *the* operation on which neural networks spend most of their time and energy. Therefore, reducing the cost of matrix multiplication is a key lever to make them more efficient. In <a href="#S4R1" class="refcite">[1]</a>, we study the efficiency of matrix multiplication on GPU with a particular kind of matrices called **Kronecker-sparse matrices**. These matrices are sparse (few nonzeros) and the locations of the nonzeros are on a support defined by Kronecker products. These matrices exhibit interesting properties that make them particularly appealing to use in neural networks. First, they can be multiplied by vectors in sub-quadratic time in theory. Moreover, their sparsity is structured (i.e., the locations of the nonzeros are not arbitrary, they must respect a Kronecker structure), so GPU implementations can potentially exploit this structure known *a priori* to transform this theoretical speedup into a practical one. For instance, the efficiency of well-known algorithms in signal processing (e.g., Fast Fourier Transform, Fast Hadamard Transform) precisely relies on computing products with Kronecker-sparse matrices (either implicitly or explicitly). Finally, empirically, imposing matrices to be Kronecker-sparse in neural networks, and training from scratch or fine-tuning, can yield models that match the accuracy of dense models in various settings (e.g., image classification with ResNets, language modeling with Transformers).

Despite these appealing properties, Kronecker-sparse matrices are currently not widely used in practice. We have not been able to find a lot of evidence about their practical efficiency (in terms of time and energy) in the literature. This is why we decided in <a href="#S4R1" class="refcite">[1]</a> to extensively benchmark existing baseline implementations for matrix dimensions (and sparsity count) typically used in Transformers. Our findings revealed that up to 50% of computation time is spent on memory operations (rather than doing actual arithmetic operations). To address this bottleneck, we have proposed a novel CUDA kernel which performs the multiplication between a dense matrix and a Kronecker-sparse matrix with a different "tiling strategy" (essentially different memory access patterns and repartition of the work between parallel threads). This new kernel reduces data transfers between the different GPU memory levels, which is critical to diminish the time and energy spent on memory operations. We find that this new kernel accelerates inference (typically by 40% on our benchmark) and reduces energy consumption (typically by 15%), especially in float32 and batch-size-last <a href="#S4R1" class="refcite">[1]</a>. This is not completely satisfying since the standard framework for deep learning (PyTorch) uses batch-size-first by default, and computations in float16 are preferred for efficiency.

The reduced efficiency in batch-size-first layouts stems from our tiling strategy, whose memory accesses are contiguous only when the batch dimension comes last. This limitation is specific to the proposed kernel design, not to Kronecker-sparse matrices themselves. 
(1) Even in batch-size-first mode, Kronecker-sparse matrices remain much more efficient than dense ones—both with existing baselines and with our kernel, which still improves performance in many setups. 
(2) Moreover, batch-size-last layouts could become increasingly common: for instance, the default sparse matrix multiplication in PyTorch is already about 10× faster in batch-size-last, suggesting that wider adoption may follow as such evidence accumulates.
As for float16, our kernel currently improves over baselines in fewer cases than in float32, but this appears to be an implementation rather than a fundamental limitation. Exploring whether further engineering around our tiling strategy can extend these gains to float16 remains an exciting open direction. 

**Takeaway.** Structure matters, exploiting it can speed up inference on GPU.
  </div>
</details>
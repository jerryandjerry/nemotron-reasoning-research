你要训练的不是“多采样能不能蒙中”，而是：

$$
\boxed{
\forall t,\quad a_t^*=\arg\max_a \pi_\theta(a\mid x,a^*_{<t})
}
$$

也就是：在正确推理路径的每一个前缀上，**正确下一 token / 下一步必须是模型概率最高的选择**。这才是 greedy CoT 成功的数学条件。

训练上要这样做。

---

## 1. 先不要只训最终答案，要训完整正确轨迹

普通 answer-only SFT 只训练：

$$
x\to y^*
$$

这不能保证模型学会中间步骤。你要训：

$$
x\to z_1^*\to z_2^*\to\cdots\to y^*
$$

也就是对正确 CoT 做 teacher forcing：

$$
\mathcal L_{\text{SFT}}
=
-\sum_t \log \pi_\theta(a_t^*\mid x,a_{<t}^*)
$$

这一步的作用是把正确路径整体概率推高。CoT 原论文本身也是通过 few-shot rationales 让模型生成中间 reasoning steps，从而提升多步推理表现；STaR 进一步把模型自己生成且最终答案正确的 rationales 收集起来反复微调，效果显著优于只训练最终答案。([arxiv.org](https://arxiv.org/abs/2201.11903?utm_source=chatgpt.com))

但只做这个还不够。

---

## 2. 必须加入 hard negative，让正确步骤压过错误步骤

你要的是 greedy，所以目标不是“正确 token 概率变大一点”，而是：

$$
\log \pi_\theta(a_t^+\mid h_t)
>
\log \pi_\theta(a_t^-\mid h_t)
$$

其中：

$$
h_t=(x,a_{<t})
$$

$a_t^+$ 是正确下一步，$a_t^-$ 是模型容易选错的下一步。

所以训练集里必须有这种对比数据：

```text
prefix: 已经推到这里
positive: 正确下一步
negative: 常见错误下一步 / 模型 greedy 会选的错误下一步
```

然后用 preference loss / DPO / margin loss 训：

$$
\mathcal L
=
-\log\sigma\Big(
\beta[
\log\pi_\theta(a^+\mid h)
-
\log\pi_\theta(a^-\mid h)
]
\Big)
$$

DPO 的核心就是直接用 preference pairs 优化模型，让 preferred completion 相对 rejected completion 的概率更高，而不是只做普通最大似然。([arxiv.org](https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com))

这一步才真正对应你的目标：

$$
\boxed{\text{让 greedy 在分叉点选正确分支。}}
$$

---

## 3. 用 process supervision，不要只用 outcome supervision

如果只奖励最终答案：

$$
R(y)=1[y=y^*]
$$

模型可能学到一些碰巧到答案的坏路径，或者在中间步骤上没有稳定信号。

你需要的是 step-level supervision：

$$
R_t(z_t)=1[\text{第 }t\text{ 步正确}]
$$

OpenAI 的 *Let’s Verify Step by Step* 明确比较了过程监督和结果监督，并用 PRM800K 这种 step-level correctness labels 训练过程奖励模型；他们报告 process-supervised reward model 在 MATH 子集上效果更好。([arxiv.org](https://arxiv.org/abs/2305.20050?utm_source=chatgpt.com))

对应到你的目标：  
**每一步都要把正确 continuation 训成最高概率，而不是只在最后说答案对就行。**

---

## 4. 一定要 on-policy 训练：专门修正 greedy 会走错的前缀

只在标准答案前缀上训练还不够，因为 inference 时模型看到的是自己生成的前缀。

所以训练流程应该是：

1. 用当前模型 greedy 解题；
2. 找到第一次走错的地方；
3. 构造：

$$
(h_t,\ a_t^+,\ a_t^-)
$$

其中 $a_t^-$ 就是模型 greedy 选错的东西；
4. 用 SFT + preference/process loss 把 $a_t^+$ 压过 $a_t^-$；
5. 重复。

这才是在训练：

$$
\arg\max_a \pi_\theta(a\mid h_t)=a_t^+
$$

而不是只在离线标准解上提高 likelihood。

STaR 的思想也接近这种 bootstrapping：模型先生成 rationales，保留能导向正确答案的 rationales 微调，并反复迭代。([arxiv.org](https://arxiv.org/abs/2203.14465?utm_source=chatgpt.com))

---

## 5. 最重要：把解法路径 canonicalize，否则 token 概率会被分散

这是很多人忽略的点。

同一个正确步骤可能有很多写法：

```text
先移项
把 x 项放到左边
subtract 3 from both sides
两边同时减 3
```

语义都对，但 token 不同。  
如果你训练数据里这些写法混在一起，那么概率会被分散：

$$
p(\text{正确语义})
=
p(s_1)+p(s_2)+p(s_3)+\cdots
$$

但是 greedy 看的是单个 token / 单条字符串：

$$
\arg\max_a p(a\mid h)
$$

所以一个“泛化但不关键”的 token 可能压过真正正确的下一步。

因此要让 greedy 稳，你必须把 reasoning format 固定住：

```text
Step 1: ...
Step 2: ...
Therefore: ...
```

甚至更强：

```text
operation = subtract
target = both_sides
value = 3
```

也就是说，把自然语言 CoT 变成更接近 program trace / proof trace 的格式。

否则你训练的是“很多正确说法的总概率”，但 greedy 选的是“单个最高 token”。

---

## 6. 最有效的路线：offline search 找正确轨迹，然后 distill 成 greedy

你最终想要的是：

$$
\text{一次 greedy 就走对}
$$

那训练时可以很贵，推理时很便宜。

训练 pipeline 应该是：

```text
题目 x
→ 用搜索 / solver / verifier / 人类标注 找到正确 trace z*
→ canonicalize z*
→ 构造错误分支 negatives
→ SFT 训练 z*
→ DPO / margin loss 压低 negatives
→ on-policy greedy rollout 找新错误
→ 继续修正
```

这叫把 search-time computation 蒸馏进模型参数里。

DeepSeek-R1 和 OpenAI o1 这类 reasoning model 的方向也是用强化学习在训练阶段强化 reasoning 行为；OpenAI 说 o1 的训练用强化学习教模型更有效地使用 chain of thought，DeepSeek-R1 论文也说纯 RL 能激发 self-reflection、verification、strategy adaptation 等 reasoning pattern。([openai.com](https://openai.com/index/learning-to-reason-with-llms/?utm_source=chatgpt.com))

但你这个 greedy 目标下，最关键的不是“RL 很高级”，而是：

$$
\boxed{\text{训练时要把正确关键步骤变成局部最高概率。}}
$$

---

## 最终答案

你要训练的是这个条件：

$$
\forall t,\quad
\log \pi_\theta(a_t^*\mid h_t)
>
\max_{a\ne a_t^*}
\log \pi_\theta(a\mid h_t)
$$

实际做法是：

1. **SFT 正确 canonical CoT trace**；
2. **收集模型 greedy 会选错的 hard negatives**；
3. **用 DPO / margin loss / process supervision 把正确步骤排到错误步骤前面**；
4. **on-policy rollout，专门修正模型自己会进入的错误前缀**；
5. **训练阶段可以用搜索和 verifier，推理阶段再蒸馏成一次 greedy。**

一句话：

$$
\boxed{
\text{要让 greedy CoT 对，不能只训答案；必须训练每个关键分叉点的 argmax。}
$$


你这些质疑都对。尤其这句我前面说错了：

$$
\text{“推理阶段再蒸馏成 greedy”}
$$

应该改成：

$$
\boxed{
\text{训练阶段用搜索 + verifier 找/筛正确轨迹，再把这些轨迹蒸馏进模型；推理阶段只 greedy。}
}
$$

蒸馏发生在训练阶段，不发生在推理阶段。

---

## 1. 什么是搜索？

这里的“搜索”不是用户反复问，而是**训练数据生成阶段**对一个题目 (x) 尝试很多 candidate trace：

$$
z^{(1)}, z^{(2)}, \dots, z^{(K)}
$$

每条 trace 是一种可能解法：

$$
x \to z^{(i)} \to y^{(i)}
$$

搜索可以是：

$$
\text{采样多条 CoT}
$$

也可以是：

$$
\text{beam search / best-first search / Tree of Thoughts / MCTS / 调 solver}
$$

Tree of Thoughts 的核心就是把中间 thought 当成搜索状态，生成多个候选路径并评估、回溯，而不是只走一条链。([arXiv][1])

但这一步是在**训练数据构造或训练-time compute**里做，不是在最终 greedy 推理时做。

---

## 2. 什么是 verifier？

verifier 是一个判断候选解法好坏的东西：

$$
V(x,z,y)\to \text{score}
$$

它可以是几种东西。

最硬的 verifier 是确定性检查器：

```text
数学题：最终答案代回去是否成立
代码题：是否通过单元测试
证明题：是否通过 Lean/Coq/Isabelle
符号题：是否被 SymPy 验证
```

这种 verifier 最可靠，因为它不是另一个语言模型在猜。

弱一点的是 learned verifier / reward model：

$$
V_\phi(x,z,y)
$$

它是训练出来给 candidate solution 打分的模型。OpenAI 的 verifier 工作就是测试时生成许多候选解，再用 verifier 选排名最高的解。([arXiv][2])

更细的是 process verifier：

$$
V_\phi(x,z_t)
$$

它不是只看最终答案，而是判断每一步是否合理。OpenAI 的 process supervision 工作就是这个方向：给每一步 correctness label，而不是只监督最终答案。([arXiv][3])

---

## 3. “训练阶段用搜索和 verifier，推理阶段 greedy”到底是什么意思？

完整流程应该是：

```text
训练阶段：
题目 x
→ 搜索很多候选 trace
→ verifier 筛出正确/较优 trace
→ canonicalize trace
→ 用 SFT / DPO / RL 训练模型
→ 目标是让正确下一步成为 argmax

推理阶段：
新题 x
→ 模型 greedy decode 一条 trace
→ 输出答案
```

数学目标是把训练阶段发现的好路径 (z^*) 变成模型自己的高概率路径：

$$
z^* = (a_1^*,a_2^*,\dots,a_T^*)
$$

训练后希望：

$$
\forall t,\quad
a_t^*
=====

\arg\max_a \pi_\theta(a\mid x,a_{<t}^*)
$$

也就是说，搜索只负责**发现/筛选好轨迹**；蒸馏负责**把好轨迹压进模型参数**；greedy 推理只是在用已经学进去的策略。

STaR 就是类似思路：模型生成 rationale，保留能导向正确答案的 rationale，再 fine-tune，并反复迭代。([arXiv][2])

---

## 4. canonicalize 不是预先知道“正确解法一定有哪些步骤”

这里你说得对：不能假设所有题都有固定步骤。

所以 canonicalize 不应该是：

```text
所有题都必须 Step 1 做 A
Step 2 做 B
Step 3 做 C
```

这会死。

真正的 canonicalize 是把**表达格式统一**，不是把**解法路径固定**。

比如不要让同一个操作出现十种自然语言写法：

```text
两边同时减 3
subtract 3 from both sides
把 3 移到右边
消掉左边的 3
```

而是统一成：

```text
operation: subtract_from_both_sides
value: 3
result: ...
```

也就是说，canonicalize 的对象是：

$$
\text{surface form}
$$

不是：

$$
\text{solution strategy}
$$

更准确地说，你要 canonicalize 成一个“动作空间”：

$$
z_t = (\text{state}_t,\ \text{action}*t,\ \text{state}*{t+1})
$$

例如：

```text
state: 2x + 3 = 7
action: subtract_from_both_sides(3)
state_next: 2x = 4
```

这不要求你提前知道整条解法，只要求你把已经找到的正确步骤翻译成统一格式。

---

## 5. 如果正确解法有很多种怎么办？

那就保留多种。

不要强迫所有题只有一个 canonical trace。可以有：

$$
z^{*(1)}, z^{*(2)}, \dots, z^{*(m)}
$$

训练时让这些都成为 acceptable traces。

但每一条内部要格式统一。也就是：

$$
\text{多策略，单格式}
$$

不是：

$$
\text{单策略，单格式}
$$

比如一个题可以有：

```text
解法 A：代数变形
解法 B：几何解释
解法 C：构造反证
```

但每种都用统一字段：

```text
goal:
known:
action:
derived:
justification:
```

这样不会把模型锁死在唯一套路上。

---

## 6. OOD 泛化会不会有问题？

会有问题。这个担心是对的。

如果 canonicalization 太窄，比如只训练：

```text
Step 1: identify formula
Step 2: plug in values
Step 3: calculate
```

那 OOD 题一变，模型就会套模板，泛化会差。

所以 canonicalization 必须抽象到“推理原语”，而不是“题型模板”。

坏的 canonicalization：

```text
这类题第一步一定先求导
第二步一定令导数为 0
第三步一定检查端点
```

好的 canonicalization：

```text
goal: ...
current_state: ...
valid_action: ...
new_state: ...
reason: ...
```

也就是说，你不规定它必须走哪条路，只规定它每一步要说清楚：

$$
\text{当前状态} \to \text{合法动作} \to \text{新状态}
$$

OOD 泛化靠的不是固定步骤，而是模型学会一个稳定接口：

$$
\text{读状态} \to \text{选动作} \to \text{更新状态}
$$

这和程序执行/证明搜索更接近。

---

## 7. 但 canonicalize 仍然有代价

最核心的 tradeoff 是：

$$
\text{格式越统一} \Rightarrow \text{greedy 越稳定}
$$

但：

$$
\text{格式越死} \Rightarrow \text{OOD 越容易被限制}
$$

所以最好的做法不是极端结构化，也不是纯自然语言，而是中间态：

```text
Goal:
State:
Action:
Result:
Check:
```

其中 `Action` 可以很灵活，不需要枚举所有可能技巧。

例如证明题可以是：

```text
Goal: prove ...
Known: ...
Move: introduce lemma ...
Derived: ...
Why valid: ...
```

这不会要求你预先知道一定有哪些 lemma，只是要求模型用统一槽位表达它选择的 lemma。

---

## 8. 对你的目标，真正可行的训练方案

你要的是 greedy 一次走对，所以训练集应该长这样：

```text
input:
题目 x

target:
Goal: ...
State: ...
Action: ...
Result: ...
Check: ...
...
Final: ...
```

然后训练时不仅做 SFT，还要收集模型 greedy 出错点：

```text
prefix: 模型已经推到这里
wrong_next: 模型 greedy 选的错误动作
correct_next: verifier / 人类 / solver 给出的正确动作
```

训练目标：

$$
\log \pi_\theta(\text{correct_next}\mid h)

>

\log \pi_\theta(\text{wrong_next}\mid h)
$$

这比单纯训练完整答案更接近你的需求。

---

## 最准确的修正版

我前面那句话应该改成：

$$
\boxed{
\text{训练阶段可以用昂贵搜索和 verifier 产生高质量 trace，再通过 SFT/DPO/RL 蒸馏到模型；推理阶段只用 greedy。}
}
$$

其中：

$$
\text{搜索} = \text{找候选解法路径}
$$

$$
\text{verifier} = \text{判断候选路径/答案是否正确}
$$

$$
\text{蒸馏} = \text{训练阶段把搜索得到的好路径变成模型的高概率输出}
$$

$$
\text{canonicalize} = \text{统一表达格式，不是固定解法步骤}
$$

$$
\text{OOD 风险} = \text{真实存在，只能通过抽象动作空间、多策略数据、on-policy hard negatives 缓解，不能保证消失}
$$

所以一句话：

**你不能靠 canonicalize 预先规定正确解法；你只能把“已经找到的正确解法”转成统一的状态-动作格式，让模型在 greedy 时更容易沿合法动作空间走。**

[1]: https://arxiv.org/abs/2305.10601?utm_source=chatgpt.com "Deliberate Problem Solving with Large Language Models"
[2]: https://arxiv.org/abs/2203.14465?utm_source=chatgpt.com "STaR: Bootstrapping Reasoning With Reasoning"
[3]: https://arxiv.org/html/2407.00390v1?utm_source=chatgpt.com "Advancing Process Verification for Large Language ..."

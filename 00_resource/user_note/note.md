1. sft的数据（包括cot和answer）必须正确，不能为用看上去正确的数据（表面cot+正确答案）。既然这个题解不出来，那就干脆别用这个题去训练。
2. 收集错误数据（用top p，不要greedy，收集多条track）， 一个是不要走错（dpo），一个是走错了能不能走回来（增加一些迂回绕路的训练数据，从模型的inference中得到启发）。总之greedy decoding一定是on-policy问题，因为每一步都要对，那模型没有在训练中实际走过怎么行呢？
3. 解题路径做成树，可以回溯
4. Logprob trick（来自huikang的0.85 pipeline）：
   - 训练完一轮后，对每个训练样本做forward pass，记录每个token的logprob
   - 每个puzzle取min_logprob（最差token）——这代表greedy decoding最容易出错的地方
   - 按min_logprob排序，最差的puzzle做oversample（2x）进入下一轮训练
   - 每轮从头训练（RESET_WEIGHTS=True），不要resume，只传递数据分布
   - Huikang的做法：run 1 → 收集logprobs → 选出priority puzzles → run 2用oversampled数据从头训
   - 他的priority list有1249个puzzle，主要是bit_manipulation(816)和cipher(180)
   - 目标：maximize the minimum logprob，把最差的puzzle提上来
   - 也可以先跑base model的logprobs（不训练，只forward pass），了解哪些puzzle天生难
5. puzzle solver must be forward only and strictly sequential, any precomputation is leakage.
6. lora 去掉一些module 然后做inference，不就能得到一些差一点的训练trace吗？
7. -，->, -to, -left, -right 都是不同tokens，不会混淆；但和 —— 是一个token吗？训练数据要检查token collision，关键是模型的眼里看上去是怎样的。
8. 模型无法片段式的学习一个完整的知识。我想让模型学习每种符号的计算顺序，但每个样本只会提及跟那个样本相关的符号，最终模型无法从多个数据中学习到整个符号计算表
9. 同一个prompt，你不能先训练一个trace再训练另一个trace，这只会让模型混乱。这个可不是curriculum learning。
10. if you are rush in time, you cannot let an agent explore and solve the problem itself, beucae it will only try everything to "solve" the problem, but it will not stop and think about what is the problem, it will not design an experinment to understand the problem, to look at it systematically, to compare to proven solution about what is the difference. All these detours are actually critical when we approach a solution, but llm doesn't know detour, its every step is a forward. Forward doesn't solve a complicated problem.
11. model cannot learn the system from 1 or 2 samples, this not possible. If you want the model learn anything, you need to use samples to reinforce that. e.g. you introduce a exotic mecahnism, which is reasonable for human even with one sample; but for llm, if you only have 10 samples about that, it doesn't know shit. 
12. model learn how to generate "unknown" answer from "unknown operator", while there is no any "unknown answer" in the train data.
13. on-policy distiallation: 假设模型的决策和训练数据分叉了，那我们就用solver在分叉的路上进行生成，让模型在分叉路上走到终点。
14. 当你看到trajectory有偏离的时候，居然没有想过可能是underfitting。0.84再训练一个epoch后就变成了0.86.
15. warmup 注重延续，不warmup注重探索，没有孰优孰劣，要根据两次数据分布的不同和学习目标来选择。如果你觉得是新的学习，那就没有必要延续。连续两个epoch还是分别两个epoch也要看你是不是需要模型每个epoch进行重新探索。（ps：你拿到手的模型本身已经是别人深度训练过的了，所以你本身就不是重头训练。）
16. 搞不懂的是warmup的训练一开始的loss反而变化很大，而直接开始decay反而变化很小。。。
17. 小数据单epoch不要warmup也不要wsd，要让它充分训练
18. 单epoch linear decay好像比cosine好？
# LogiQA Pilot tolerant 不一致样本定性分析

## 范围与方法

- 分析范围严格限于 tolerant 结果不一致的 18 条：14 条 correct→wrong、4 条 wrong→correct。
- 答案与转移来自 predictions.jsonl 和 audit_cases.jsonl，不重新解析或改写已有审计结果。
- 这两份 JSONL 没有保存原始 query 和选项。为完成题目级核对，本报告按 question_id 只读对齐了现有 Pilot 的 summary.json 所记录的本地数据源 /tmp/logiqa2-dev.txt；没有下载、扩充或推理新数据。
- 全部判断由人工阅读题目、选项和已有 Solver/Critic/Refiner 原始输出完成。没有调用 LLM/API，也没有使用自动语义分类器。
- “生成前信号”只使用 Critic 生成前可见的题目、选项和 Solver 输出，是事后定性观察，不是控制器预测结果。
- Critic 的 CHANGE 表示明确否定 Solver 答案并要求修改；即使没有给出确定字母，只要修改立场明确，也不记为 UNCLEAR。

## 总览

| tolerant 转移 | 数量 | Critic KEEP | Critic CHANGE | Critic UNCLEAR |
|---|---:|---:|---:|---:|
| wrong→correct | 4 | 0 | 4 | 0 |
| correct→wrong | 14 | 0 | 14 | 0 |
| 合计 | 18 | 0 | 18 | 0 |

| 主要原因 | 数量 | 样本 ID |
|---|---:|---|
| helpful_critique | 4 | 2288, 7040, 11554, 7450 |
| false_critique | 12 | 8898, 13622, 6137, 7776, 10583, 6643, 14352, 1222, 1444, 8685, 4219, 6773 |
| wrong_alternative | 2 | 14532, 11202 |
| refiner_ignored_keep | 0 | — |
| refiner_failed | 0 | — |
| ambiguous | 0 | — |

## wrong→correct：4 条

### ID 2288

- **题目与选项**：材料强调环境数据造假已经形成利益链，责任不仅涉及企业和直接操作者，还涉及幕后指挥者、地方环保部门、设备厂商和运维单位。问材料意在说明什么。A“造假已系统化”；B“打击造假不能只处罚涉事企业”；C“杜绝造假必须覆盖整个链条”；D“地方环保部门为了政绩漂白数据”。
- **Gold / 答案**：Gold C；Solver B；Critic CHANGE，建议 C；Refiner C。
- **Solver 依据**：正确识别了“系统性、多方参与、需要综合治理”，但最后选择了只覆盖“不能只处罚企业”的较窄 B。
- **主要原因**：helpful_critique。Critic 指出 B 只覆盖责任链的一部分，C 才概括“整条链条”，并被 Refiner 正确执行。
- **生成前信号**：**答案与推理不一致**。Solver 自己已说材料“suggests a comprehensive approach”，但答案落在部分表述 B。
- **原始输出证据**：Solver：“The passage discusses the systemic nature ... It suggests a comprehensive approach. Option B captures ...”；Critic：“Option B is partially correct but misses the broader systemic implication ... C a more accurate reflection”；Refiner：“Option C better reflects ... the entire chain of responsibility. ... FINAL_ANSWER: C”。

### ID 7040

- **题目与选项**：材料比较阅读文字和观看电视：文字更易记忆、对照信息和发现矛盾，电视使人较少反思并依赖直觉。A“阅读文字比看电视更有助于思考”；B“信息接收方式影响人的行为”；C“电视使人形成错误价值观”；D“爱阅读者比爱看电视者更冷静”。
- **Gold / 答案**：Gold A；Solver B；Critic CHANGE，建议 A；Refiner A。
- **Solver 依据**：概括了文字在回忆、比较和批判思考上的优势，却用更泛化的“影响行为”选择 B。
- **主要原因**：helpful_critique。Critic 准确指出 B 过宽，材料核心是阅读相对于电视对思考的帮助。
- **生成前信号**：**答案与推理不一致**。Solver 的具体依据直接支持 A，而非其最终选择的概括性 B。
- **原始输出证据**：Solver：“reading allows for better recall and comparison ... less reflection ... FINAL_ANSWER: B”；Critic：“Option B is overly general ... Option A is more directly supported”；Refiner：“reading text enhances recall and critical thinking ... FINAL_ANSWER: A”。

### ID 11554

- **题目与选项**：Newtown 教职申请数比 1985 年下降，学生数和教师辞职数却上升，但 1990 年代末没有教师短缺。问最能解释矛盾的事实。A 新住宅将使小学生增加 12%；B 1993 年申请数比岗位数多 40%；C 学校董事会不考虑提高师生比；D 周边师范院校毕业生减少。
- **Gold / 答案**：Gold B；Solver C；Critic CHANGE，建议 B；Refiner B。
- **Solver 依据**：把“不提高师生比”解释为“现有教师足够”，但 C 只描述政策意图，并不提供教师供给充足的证据。
- **主要原因**：helpful_critique。Critic 找到真实缺口，并指出 B 的岗位申请盈余直接解释为何申请下降仍不短缺。
- **生成前信号**：**推理内部矛盾；未完整排除选项**。从“不提高师生比”推不出“当前教师足够”，且 Solver 没有比较直接量化供给盈余的 B。
- **原始输出证据**：Solver：“Option C ... implying that the current number of teachers is sufficient”；Critic：“Option C merely states ... which does not explain why there is no teacher shortage ... Option B ... a surplus of qualified candidates”；Refiner：“Option B explains this ... more applications than available positions ... FINAL_ANSWER: B”。

### ID 7450

- **题目与选项**：材料以莫言作品说明“只有民族的，才是世界的”。A“民族的就是世界的”；B“如果不是民族的，就不是世界的”；C“只要是民族的，就是世界的”；D“只有世界的才是民族的”。
- **Gold / 答案**：Gold B；Solver A；Critic CHANGE，建议 B；Refiner B。
- **Solver 依据**：识别了原命题，却把“只有 P 才 Q”的必要条件关系错误改成“P→Q”，选择了逆命题 A。
- **主要原因**：helpful_critique。Critic 正确指出 B 是原命题的逆否命题，而 A 是不能推出的逆命题。
- **生成前信号**：**答案与推理不一致**。Solver 引用了“only the national is the world”，但所选 A 与该逻辑结构不一致。
- **原始输出证据**：Solver：“aligning with the statement ‘only the national is the world.’ ... Option A ... directly reflects this idea”；Critic：“Option A ... is a converse ... Option B ... is the contrapositive”；Refiner：“Option B ... is the contrapositive and logically valid. ... FINAL_ANSWER: B”。

## correct→wrong：14 条

### ID 8898

- **题目与选项**：新商品房每平方米均价上涨 25%，平均家庭预算中购房支出占比不变，据此推出家庭预算也上涨 25%。问必要假设。A 平均家庭购买面积不变；B 食品教育占比不变；C 全国购房总额上涨 25%；D 所有住房相关支出占比不变。
- **Gold / 答案**：Gold A；Solver A；Critic CHANGE，建议 D；Refiner D。
- **Solver 依据**：总购房成本等于单价乘面积；要从每平方米价格上涨推出购房成本同比上涨，必须固定购买面积。
- **主要原因**：false_critique。Critic 把题干已经给出的“购买商品房支出占预算比例不变”误当成待补假设，并用范围更宽的“所有住房相关支出”替代真正缺失的面积条件。
- **生成前信号**：**无明显信号**。Solver 明确指出数量必须保持不变，答案与必要假设一致。
- **原始输出证据**：Solver：“requires that the quantity of housing purchased remains constant ... FINAL_ANSWER: A”；Critic：“assumption should be broader ... Option D ... more comprehensive”；Refiner：“best captured by D ... FINAL_ANSWER: D”。

### ID 13622

- **题目与选项**：Tony 认为人生人格在线性经历中发展，小说才能准确描绘；Raoul 认为人生是互不连贯的片段，短篇小说更忠实。问二人对什么存在分歧。A 短篇能否揭示隐藏性格；B 是否只有短篇是小说家的草稿；C 小说和短篇是否使用同样策略；D 人生是否最好理解为互不连贯的片段。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，未给确定字母，只要求改成“哪种媒介更能反映人生”；Refiner C。
- **Solver 依据**：对照 Tony 的连续人格发展观和 Raoul 的离散片段观，定位到对人生结构的直接分歧。
- **主要原因**：false_critique。Critic 声称 Tony 没有反对“人生由离散片段组成”，与 Tony 明示的渐进人格发展叙述冲突；C 的“采用相同策略”也不是双方争点。
- **生成前信号**：**无明显信号**。Solver 对双方主张的对照完整，D 与对照一致。
- **原始输出证据**：Solver：“Tony ... character development, while Raoul ... disjointed vignettes ... FINAL_ANSWER: D”；Critic：“Tony does not dispute that life consists of disjointed vignettes”；Refiner：“Their disagreement centers on which medium ... FINAL_ANSWER: C”。

### ID 6137

- **题目与选项**：材料介绍用激光冷却原子，利用原子对地磁和重力场的敏感性进行定位，并提到未来需微型化以装入手机。A 智能手机应用创新；B 激光的新用途；C 新定位系统在前沿领域的应用；D 利用原子定位的原理。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，建议 C；Refiner C。
- **Solver 依据**：抓住主要篇幅所解释的“冷却原子—感知场变化—实现定位”的工作原理。
- **主要原因**：false_critique。Critic 过度放大末尾的潜在手机用途；C 所说“前沿领域应用”并非材料主体，且材料把手机化描述为尚待解决的问题。
- **生成前信号**：**无明显信号**。Solver 的主旨概括覆盖材料核心机制。
- **原始输出证据**：Solver：“explains how atoms ... are sensitive ... enabling precise positioning ... FINAL_ANSWER: D”；Critic：“Option C better reflects the application in a real-world context”；Refiner：“C better captures ... practical use. ... FINAL_ANSWER: C”。

### ID 7776

- **题目与选项**：定义把垄断描述为一个企业或少数大企业共同控制相应部门产品的生产和销售。A 连锁餐厅全国统一定价；B 几个主要生产商约定共同维持价格；C 跨国公司的某产品都在中国工厂生产；D 政府限制新建工厂。
- **Gold / 答案**：Gold B；Solver B；Critic CHANGE，没有明确推荐答案；称 B 只是合谋，并提到 C 可能暗示垄断但缺少支配证据；Refiner C。
- **Solver 依据**：B 中“几个主要生产商共同维持价格”与题干“少数大企业共同控制销售”相符。
- **主要原因**：false_critique。Critic 引入了题干之外的“合谋/寡头不算垄断”窄定义，忽略题干明确允许“少数企业共同控制”；其自己也承认 C 缺少市场支配证据。
- **生成前信号**：**无明显信号**。Solver 直接按题干定义匹配实例。
- **原始输出证据**：Solver：“several major producers ... aligns with ... joint control. FINAL_ANSWER: B”；Critic：“Option B describes a price-fixing agreement ... not necessarily monopoly ... Option C could imply monopoly ... but it lacks explicit control claims”；Refiner：“Option C suggests ... FINAL_ANSWER: C”。

### ID 10583

- **题目与选项**：环保人士重视航天卫星带来的环境监测收益，却没有考虑航天器可能严重破坏臭氧层。问最符合的原则。A 人们容易忽视支持自身活动的行动所带来的不良后果；B 使用技术常有未预见的负面后果；C 技术通常至少对环境有些负面影响；D 巨大正面效果可抵消负面效果。
- **Gold / 答案**：Gold A；Solver A；Critic CHANGE，建议 B；Refiner B。
- **Solver 依据**：材料的关键不是一般技术风险，而是环保人士因卫星支持其环保活动而忽视反面后果。
- **主要原因**：false_critique。Critic 把“有动机地忽视支持自身活动的负面后果”改写成一般性的“技术存在未预见后果”，丢失了论证的主体偏向。
- **生成前信号**：**无明显信号**。Solver 准确对应了行动支持自身活动与忽视后果两个要素。
- **原始输出证据**：Solver：“overlook potential negative consequences ... despite their benefits ... FINAL_ANSWER: A”；Critic：“Option B is more broadly applicable ... unforeseen negative outcomes”；Refiner：“Option B better captures ... FINAL_ANSWER: B”。

### ID 14532

- **题目与选项**：顾问仅凭本季销量下降、新产品尤其差，就断定竞争顾问建议的广告活动设计不当。问论证最易受何种批评。A 混淆必要与充分条件；B 默认没有该广告销量不会更低；C 未考虑与广告无关的经济因素导致低销量；D 默认新品应超过老产品。
- **Gold / 答案**：Gold C；Solver C；Critic CHANGE，建议 B；Refiner B。
- **Solver 依据**：明确识别出从低销量归因到广告的因果缺口，并指出可能存在无关经济因素。
- **主要原因**：wrong_alternative。Critic 承认 C 有效、也识别到因果推断需要排除替代解释，却错误地把只讨论“没有广告时是否更差”的 B 排在覆盖外部原因的 C 之前。
- **生成前信号**：**无明显信号**。Solver 的答案和因果混淆说明一致。
- **原始输出证据**：Solver：“overlooks the possibility that external economic factors ... FINAL_ANSWER: C”；Critic：“correctly identifies ... aligns with option C. However ... B presents a more direct flaw”；Refiner：“fails to rule out other explanations ... FINAL_ANSWER: B”。

### ID 11202

- **题目与选项**：三挂车事故率低于单、双挂车，据此建议增加三挂车使用以减少死亡。问最强削弱。A 很少有致命卡车事故是两卡车相撞；B 一些小路始终禁行大型卡车；C 三挂车迄今只在车流稀少的主要公路路段使用；D 行业安全记录略有改善。
- **Gold / 答案**：Gold C；Solver C；Critic CHANGE，建议 A；Refiner A。
- **Solver 依据**：选中了正确的选择偏差选项，但把其作用表述为“当前使用有限，因此影响不大”，没有清楚说出低事故率可能来自轻交通道路而非车型。
- **主要原因**：wrong_alternative。Critic 确实观察到 Solver 对 C 的解释偏离了论证要点，但没有补上“道路环境混杂”这一正确理由，反而推荐与单车/多车碰撞无关的 A。
- **生成前信号**：**推理内部矛盾**。结论建议“增加使用”，所以“现在用得少”本身并不削弱；真正应检查的是现有事故率能否外推。
- **原始输出证据**：Solver：“only used on lightly traveled ... limited usage may not significantly impact ... FINAL_ANSWER: C”；Critic：“reasoning is slightly off ... conclusion is about increasing use ... Option A is more relevant”；Refiner：“Option A ... truck type may not significantly affect ... FINAL_ANSWER: A”。

### ID 6643

- **题目与选项**：材料批评婚姻中的金钱崇拜、父母干预，以及以家庭、学历和工作为主要参考，最后追问“金钱能决定一切吗？当然不能”。A 只基于美貌激情的婚姻会失败；B 感情最重要而非经济条件；C 女性应借婚姻提升地位；D 好婚姻需要正确婚姻观和态度。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，建议 B；Refiner B。
- **Solver 依据**：把对单一物质标准和外部干预的批评概括为需要正确、全面的婚姻观。
- **主要原因**：false_critique。材料只否定“金钱决定一切”，没有推出“感情是最重要因素”；Critic 用更具体但未被文本支持的 B 替换了综合结论 D。
- **生成前信号**：**无明显信号**。Solver 没有把“非金钱因素”擅自限定成某一个最重要因素。
- **原始输出证据**：Solver：“implying that other factors are also important ... proper understanding and attitude ... FINAL_ANSWER: D”；Critic：“D ... reasonable but ... Option B is more directly supported”；Refiner：“B more directly reflects ... FINAL_ANSWER: B”。

### ID 14352

- **题目与选项**：对流层温度随高度下降；平流层从对流层顶开始，温度停止下降并随高度上升。问必然为真。A 极地对流层更厚；B 任一点的平流层顶部温度至少与其正下方对流层顶部一样高；C 北极平流层中部至少和赤道一样高；D 极地平流层顶比赤道暖。
- **Gold / 答案**：Gold B；Solver B；Critic CHANGE，未给确定字母；Refiner C。
- **Solver 依据**：由同一垂直位置上平流层温度随高度上升，推出平流层顶部不低于其底部/对流层顶部。
- **主要原因**：false_critique。Critic 把 B 误读成“平流层每一点都需要精确温度”，但 B 只比较同一位置的顶部与下方边界，单调上升已经足够；C 的跨纬度比较没有依据。
- **生成前信号**：**无明显信号**。Solver 的单调性推导直接且与 B 一致。
- **原始输出证据**：Solver：“stratosphere's temperature increases with altitude ... top ... warmer ... FINAL_ANSWER: B”；Critic：“Option B makes a universal claim that cannot be definitively supported”；Refiner：“Option C ... more directly supported ... FINAL_ANSWER: C”。

### ID 1222

- **题目与选项**：材料先述大众创业创新改革和创业主体大众化，继而强调机制制度缺陷妨碍人的价值实现，长期不保障人的价值会压制人才和科技成果。A 改革应创造制度环境；B 创业主体将从精英走向大众；C 机制制度妨碍人的价值；D 保障人的价值是大众创业创新的前提。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，建议 A；Refiner A。
- **Solver 依据**：抓住后半段反复展开的条件链：人的价值若长期不能保障，创新创业人才和成果就不能释放。
- **主要原因**：false_critique。Critic 回到开头背景句 A，忽略材料主体用反事实后果强调“保障人的价值”这一前提。
- **生成前信号**：**无明显信号**。Solver 的概括与材料重点及条件链一致。
- **原始输出证据**：Solver：“obstacles to realizing human value ... affects ... innovation ... FINAL_ANSWER: D”；Critic：“Option D is overly broad ... Option A ... reflects the reform's purpose”；Refiner：“emphasis lies on the reform's role ... FINAL_ANSWER: A”。

### ID 1444

- **题目与选项**：材料说明美国只有部分重型低附加值装配因物流回流，而高价值部件仍在海外；新兴国家经多年转移形成规模和比较优势，对美国制造业回流构成挑战。A 美国制造业反弹严重影响新兴国家；B 提升技术附加值有利制造业；C 国际分工变化给美国回流机会；D 美国制造业与新兴市场竞争不具整体优势。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，建议 C；Refiner C。
- **Solver 依据**：区分了局部回流与整体竞争格局，重点概括新兴国家既有优势对美国回流的制约。
- **主要原因**：false_critique。Critic 把开头“个别公司部分装配回流”的背景现象提升为主旨 C，忽略结尾明确落在新兴国家优势和美国面临挑战。
- **生成前信号**：**无明显信号**。Solver 的主旨与材料结论一致。
- **原始输出证据**：Solver：“main focus is on the challenges ... established advantages of emerging countries ... FINAL_ANSWER: D”；Critic：“Option C better captures ... change in international division of labor”；Refiner 在重复“main focus ... challenges”后仍输出“FINAL_ANSWER: C”。

### ID 8685

- **题目与选项**：五个车站自西向东；Fu Yi 在 Hao Yun 以东、Hu Yao 以西，并与 Hu Yao 相邻；Jiu Shang 与 Yin Ling 相邻。选可能顺序。A Yin–Hao–Jiu–Fu–Hu；B Fu–Hu–Jiu–Yin–Hao；C Hao–Yin–Jiu–Fu–Hu；D Hao–Hu–Fu–Yin–Jiu。
- **Gold / 答案**：Gold C；Solver C；Critic CHANGE，建议 D；Refiner D。
- **Solver 依据**：逐项检查 C：Hao 在 Fu 西侧、Fu 紧邻且在 Hu 西侧、Yin 与 Jiu 相邻。
- **主要原因**：false_critique。Critic 读错了选项 C，声称 Yin 和 Jiu 位于第二、第三却“不相邻”；同时 D 把顺序写成 Hu–Fu，违反 Fu 必须在 Hu 西侧。
- **生成前信号**：**无明显信号**。Solver 列出的 C 顺序完整满足全部约束。
- **原始输出证据**：Solver：“Option C satisfies all conditions: Hao Yun, Yin Ling, Jiu Shang, Fu Yi, Hu Yao”；Critic：“Jiu Shang is third and Yin Ling is second, which are not adjacent ... Option D satisfies”；Refiner：“Option D satisfies all conditions ... FINAL_ANSWER: D”。

### ID 4219

- **题目与选项**：若语言完全有效，则每种基本语言组合都能成为有独立意义的词；若听觉系统接收声音信号有问题，则并非每种组合都能成为独立词。A 听觉正常即可推出每种组合都成词；B 语言有效导致交流实用；C 每种组合成词即可推出语言完全有效；D 听觉接收有问题则语言不能完全有效。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，建议 A；Refiner A。
- **Solver 依据**：用“语言完全有效→所有组合成词”和“听觉故障→并非所有组合成词”作逆否/冲突推导，得到听觉故障→语言不完全有效。
- **主要原因**：false_critique。Critic 反而把“听觉正常”错误当成“所有组合成词”的充分条件，完成了原文不支持的逆命题。
- **生成前信号**：**无明显信号**。Solver 的条件推导与 D 一致。
- **原始输出证据**：Solver：“auditory system functionality is necessary for a language to be fully effective ... FINAL_ANSWER: D”；Critic：“Option D misrepresents ... A ... aligns more closely”；Refiner：“Option A correctly reflects this ... FINAL_ANSWER: A”。

### ID 6773

- **题目与选项**：材料承认人口迁移不一定使迁入地发展，但断言从历史看“任何发达地区必然是人口迁移的结果”，并举古希腊、英伦和东北为例。A 区域间流动人口就是迁移人口；B 古代中国人口迁移受限制；C 不应歧视迁移者；D 没有人口迁入就没有区域发展。
- **Gold / 答案**：Gold D；Solver D；Critic CHANGE，没有明确建议字母，只称 B 部分得到支持；Refiner B。
- **Solver 依据**：把“任何发达地区都必须以人口迁移为结果”写成其等价的必要条件表达 D。
- **主要原因**：false_critique。Critic 把材料明确的必要条件误降为“强相关而非严格因果”；B 只截取开头“乡土观念限制流动”的一半，遗漏同句的“人口流动又是自由的”。
- **生成前信号**：**无明显信号**。Solver 的答案直接对应材料的“must”与“inseparable”表述。
- **原始输出证据**：Solver：“any developed area must be the result of population migration ... FINAL_ANSWER: D”；Critic：“option D overstates ... strong correlation but not a strict causation. Option B is also partially supported”；Refiner：“Option B is partially supported ... FINAL_ANSWER: B”。

## 汇总判断

### 1. 4 个 corrected 样本是否有共同生成前信号

四条都存在可由题目、选项和 Solver 输出本身检查的**推理到选项的贴合缺口**，但没有一个细分类标签严格覆盖全部四条：

- 2288、7040、7450：明显的“答案与推理不一致”；Solver 的文字依据分别支持更全面的 C、更具体的 A、以及必要条件的逆否 B。
- 11554：从“不提高师生比”跳到“教师供应足够”，属于推理内部缺口，并且没有排除直接给出供给盈余的 B。

因此，可以说四条都有可观察的生成前缺口；不能说存在一个已经被验证、可以直接部署的统一控制信号。这只是 4 条事后定性观察。

### 2. 14 个 degraded 样本主要由 Critic 还是 Refiner 导致

**主要由 Critic 导致。**

- 14 条中 Critic 全部采取 CHANGE，没有一条建议 KEEP。
- 其中 10 条 Critic 直接给出错误替代字母；另外 4 条（13622、7776、14352、6773）虽未给出无歧义字母，但已经错误否定 Solver，Refiner 随后补全了错误答案。
- 12 条属于对正确答案或正确依据的 false_critique；2 条（14532、11202）观察到局部问题或替代论证空间，但最终推荐了 wrong_alternative。
- 没有出现“Critic 建议保留但 Refiner 修改”，也没有出现“Critic 给出正确修改意见但 Refiner 执行失败”，所以 refiner_ignored_keep 和 refiner_failed 都是 0。
- Refiner 的次要问题是缺少对 Critic 的反证检查：它在 14 条中都接受了 Critic 的否定方向，尤其在 Critic 未明确给字母的 4 条中仍自行落到错误选项。

作为对照，14 条 degraded 中只有 11202 的 Solver 输出呈现明显的生成前推理缺口，其余 13 条没有明显矛盾、不确定表达或答案—推理错位。当前 Critic 因此表现出明显的过度修改倾向。

### 3. 建议

**选择 B：先修复 Critic/Refiner prompt。**

依据不是单纯比较 gold，而是原始输出中可复核的行为模式：Full 在这 18 个不一致样本里只修正 4 条，却破坏 14 条；Critic 对全部 18 条都选择 CHANGE，且多数退化来自误读条件关系、忽略题干给定定义、过度放大次要信息或读错选项。Refiner 又缺少在改答前验证 Critic 指控的机制。

在开始采集控制器训练数据前，应先降低 Critic 的默认反驳倾向，并要求 Refiner 只有在 Critic 提供可由题干和选项验证的反证时才改答。否则，当前工作流产生的动作标签会混入大量“错误 Critic 导致改答”的噪声。

# E1 实体对齐报告（entity_alignment.md）

> 依据：protocol_frozen.md §5/§6 对齐规则（规范化精确匹配、丢弃强制计数）| 生成：2026-08-22
> CbG 侧对齐率已定，活性证据表待 ChEMBL 抓取完成后回填（见 §2 待回填标记）。

## 1. CtD（ClinicalTrials.gov → Hetionet Compound × Disease）

### 1.1 抓取与对齐率（源：`ctgov_alignment_report.json`）

| 项 | 数值 | 说明 |
|---|---|---|
| 查询病种数 | 137 | Hetionet 全部 Disease 节点（实际 137 个，含 3 个常用别名变体） |
| 命中研究（startDate > 2017-01-01） | 186,670 | `query.cond` 逐病种 + `filter.advanced=AREA[StartDate]RANGE[2017-01-01,MAX]` |
| condition 字符串总数 / 对齐 | 567,325 / **83,494（14.7%）** | 未对齐=超出 137 病种宇宙（多数研究针对其他疾病），如实丢弃 |
| intervention 字符串总数 / 对齐 | 339,381 / **29,356（8.7%）** | 未对齐=非 Hetionet 1,552 化合物（安慰剂/器械/其他药物/组合词），如实丢弃 |
| 对齐后证据对 | **1,521** (compound × disease) | 8,758 次对齐发射去重 |

### 1.2 循环防护量化（已程序验证）

1,521 证据对三分解（对照 `build_core_dataset(42)` 全边集）：
- **241 对 = KG 已知 CtD 边**（2017 前已知适应症的试验再确认）——循环对，**构造上不在候选空间**，自动排除 ✓
- **1,280 对 = CtD absent 对**（进入富集检验的全体；其中 42 对仅有已知 CpD（palliates）边而 CtD 缺失——对称保留，已在协议语义内）
- 其余 = 上述两类的并集核对（283 对有 CtD 或 CpD 已知边，与上两项之交已对账）

### 1.3 时间窗分布

证据对按试验开始年分层于 `evidence_ctd.tsv` 的 `start_years` 列；W1(2017-2020)/W2(2021-2026) 两窗命中见结果 JSON。

## 2. CbG（ChEMBL → Hetionet Compound × Gene）

| 项 | 数值 | 说明 |
|---|---|---|
| 化合物 InChIKey 覆盖 | **1,552/1,552** | hetionet-v1.0.json.bz2（Git LFS）逐化合物提取 |
| InChIKey → ChEMBL 分子 | **1,353/1,552（87.2%）** | 双下划线过滤器 + 严格回验；199 未命中=该结构不在 ChEMBL（老药/复方/withdrawn），如实丢弃 |
| 单蛋白靶点 | 11,055 | target_type=SINGLE PROTEIN 全量 |
| UniProt accession → 基因符号 | 10,420 | rest.uniprot.org gene_primary |
| 靶点 → Hetionet 基因（精确符号匹配） | **6,265** | 多符号靶点（罕见）丢弃计数 |
| 活性证据对 | **1,357**（自 45,763 条 ≤100nM 活性中的 5,531 条 2018+ 记录） | relation=`=`、type∈{IC50,Ki,Kd,EC50,AC50,Potency}、≤100nM、document_year>2017（最终版 2026-08-23 完成） |

## 3. 对齐方法的已知局限（如实入论文 Limitations 素材）

1. CtD 病种/药物对齐为**规范化精确匹配**（无模糊检索）——保守：真实可对齐但写法不同的条件/干预被计为丢弃（方向对称，稀释而非偏置）
2. 试验的 interventions×conditions 笛卡尔积可能引入非因果组合（标准 repurposing 验证惯例，Zhang 2021 JBI 同法）
3. ChEMBL document_year 是文献发表年代（非实验执行年代）——冻结定义为"公开证据出现时间"，两窗对称
4. CpD（palliates）声明不可外部验证（协议 §7）：试验注册不区分 treat/palliate 意图

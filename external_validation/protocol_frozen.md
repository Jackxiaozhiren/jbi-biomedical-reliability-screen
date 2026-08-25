# E1 时间切片外部验证 — 冻结协议（protocol_frozen.md）

> **状态：FROZEN** | 冻结时间：2026-08-22 | 冻结人：JBI 转投流程编排（用户 2026-08-22 授权"按你的思路进行"）
> 依据：Master Prompt v1 §四 E1 + §六 Phase 2；`ccf-experiment-designer` 设计契约；W4/W5 联网与 API 现场验证（2026-08-22，见 §10）
> **铁律绑定：本文档定稿后，主结果跑出前不得修改任何阈值/时间窗/对齐规则。修改=重新冻结并留痕。**
> **主结果跑出前禁止查看任何 KEEP/WITHHOLD × 外部证据交叉数字**（防钓鱼：阈值已全部先行冻结于 §3，无后调空间）。

---

## 1. 科学问题与可检验主张

**问题**：Hetionet 快照（2017 年发表，边汇集自更早来源）之后出现的真实世界证据，是否富集于可靠性筛查层保留（KEEP）的候选预测集？

**可检验主张 H1**：在校准操作点（target 0.10），KEEP 的 absent 候选对被"快照后新证据"命中的比率显著高于未保留（WITHHOLD）候选对。
**H0**：两命中率无差异。阴性结果=结果而非失败（如实报告并触发 D2 回退，Master Prompt 铁律 4）。

## 2. 设计概要与一项对 Master Prompt 的设计修正

**Master Prompt 原表述**为池内比较（KEEP vs WITHHOLD 于 J=9 审计池内）。执行期发现：主审计池为**类型无约束**构造（负尾从全 10,526 实体宇宙抽取），CtD 查询的负候选中 Disease 类型期望仅 ~17 个——池内药物→疾病 absent 候选量级不足以支撑富集检验。**冻结修正**（功能等价、统计功效更高）：

> **把 §3 冻结阈值应用于全量类型合法 absent 候选空间**，产生部署意义上的 KEEP/WITHHOLD 全集，在其上做外部富集检验。对照逻辑与 Master Prompt 一致（KEEP vs WITHHOLD vs 随机等大集）。

此修正已按"发现事实性偏差→报告→不静默偏离"原则登记（`qa/02_实验升级记录.md` 同步）。

## 3. 冻结的操作点与阈值（全部取自 `results/hetionet_audit_J9_K500.json`，主审计模型 RotatE）

| 操作点 | 阈值 τ（index 上界，p≤τ 即 KEEP） | 冻结值 | 角色 |
|---|---|---|---|
| **全局校准 target 0.10** | calibrated_cutoff | **0.00399202** | **主操作点**（CtD 与 CbG 共用） |
| 全局校准 target 0.20 | calibrated_cutoff | 0.0179641 | 敏感性 |
| 代价感知 1:1 | threshold | 0.0199601 | 敏感性 |
| 逐关系 CtD（0.10） | cutoff | 0.0179641 | 敏感性 |
| 逐关系 CbG（0.10） | cutoff | 0.00598802 | 敏感性 |

名义 BH 在 Hetionet 拒绝零个（vacuous），不参与外部验证（论文如实注明）。

## 4. 候选空间构建（冻结）

- **CtD 空间**：全部 (Compound, CtD, Disease) absent 三元组 = {compound}×{disease} − 全 KG 已知 CtD 边（train/valid/test 全并集）。实体全集来自 `build_core_dataset(42)` 的 ID 映射（Compound::DBxxxxx，Disease::DOID:xxxx）。
- **CbG 空间**：全部 (Compound, CbG, Gene) absent 三元组（同上构造；Gene 宇宙 ~8.8k）。约 1,550×8,868−11,327(已知 CbG 边，以脚本实数为准) ≈ 13.6M 对——批量打分可行（CPU，分 query 处理）。
- **index 计算**：与主审计完全同式（Eq.1）：每查询 (h, CtD/CbG) 抽 K=500 参考尾（排除该查询全部已知正尾与候选尾，`default_rng(seed)` 独立种子流 `rng_ev=42`，与主审计采样流分离并记录），p=(1+#{ref≥cand})/(K+1)。
- **KEEP/WITHHOLD**：p≤τ → KEEP；否则 WITHHOLD。主操作点 τ=0.00399202。
- **模型**：主=RotatE（`models/hetionet_core_RotatE.pt`）；敏感性=ComplEx（E2 训成后补跑，不阻塞主结果）。

## 5. CtD 外部真值（ClinicalTrials.gov API v2，无需密钥）— 冻结抓取与对齐规则

**抓取语法（2026-08-22 实测通过）**：
```
GET https://clinicaltrials.gov/api/v2/studies
    ?query.cond=<disease label>
    &filter.advanced=AREA[StartDate]RANGE[2017-01-01,MAX]
    &pageSize=1000&pageToken=<token>
```
- 对 Hetionet 全部 **134 个 Disease 标签**逐一查询（`query.cond` 的检索扩展由 API 承担）；**不使用 `fields` 参数**（实测部分字段路径触发空响应，取全量记录客户端解析）。
- 从每条 study 提取：`statusModule.startDateStruct.date`、`conditionsModule.conditions[]`、`armsInterventionsModule.interventions[].name`（仅 type=DRUG 或名称可匹配药物的条目）。

**对齐规则（冻结，对称适用于 KEEP 与 WITHHOLD）**：
1. **Disease 对齐**：trial condition 规范化（小写、去标点、单复数归一、去修饰词 "disease/syndrome/chronic"）后与 134 个 Disease 标签规范化形式**精确匹配**；一对多时保留全部可匹配 disease（保守扩大分母，稀释方向对称）。对齐率 = 匹配到 ≥1 Disease 的 condition 数 / 总 condition 数。**未匹配 condition 的 trial 不产生对**（如实计数丢弃）。
2. **Compound 对齐**：intervention name 规范化后与 Hetionet Compound name（含 dhimmel `compounds.tsv` 的 synonym 列若有）精确匹配；不模糊匹配。未匹配不产生对。
3. **证据对定义**：(compound, disease) ∈ 同一 study 的 interventions×conditions 笛卡尔积，且 startDate > 2017-01-01。
4. **防污染**：证据对若对应 KG 已知 CtD/CpD 边（即非 absent）自动不在候选空间内（空间构造已排除）；无需额外剔除。** trials whose conditions 全部无法对齐 → 计入对齐报告，不静默丢弃。

**时间窗分层（冻结）**：W1=2017-01-01–2020-12-31；W2=2021-01-01–2026-08-22（按 startDate 归层）。

## 6. CbG 外部真值（ChEMBL webresource，免费）— 冻结抓取与对齐规则

**工程教训（2026-08-22 实测）**：`activity_year__gte` 是**静默无效过滤器**（返回 200 但不过滤——order_by=activity_id 首条 document_year=2004 实锤）。`fields` 参数在部分端点也被忽略。**日期过滤一律用响应内嵌 `document_year` 客户端后过滤**。

**对齐（冻结）**：
1. Compound：dhimmel `hetionet/nodes.tsv`+`compounds.tsv`（GitHub raw，公开）取 InChIKey → `GET /molecule.json?molecule_structures.standard_inchi_key=<ik>&limit=1` → chembl_id。多对一取首个；无匹配记丢弃。
2. Gene：ChEMBL `target.json?target_type=SINGLE PROTEIN`（分页全量）→ `target_components[].accession`（UniProt）→ UniProt REST idmapping（人）→ gene symbol → 与 Hetionet Gene 标签精确匹配。备选（idmapping 失败时）：`target_component_synonyms` 含 symbol 精确匹配。对齐率两步分别报告。
3. **Activities**：对映射成功的 (chembl_id 集 × target 集) 分批 `GET /activity.json?molecule_chembl_id__in=<batch>&target_chembl_id__in=<batch>&standard_type__in=IC50,Ki,Kd,EC50,AC50,Potency&standard_value__lte=100&standard_units=NANOMOLAR&standard_relation=EXACT`（过滤参数以脚本冒烟实测为准，超长 URL 分批 ≤50 id/批）。**效力分档（冻结）**：Tier-1 = standard_value ≤ **10 nM**（高亲和结合）；Tier-2 = ≤ **100 nM**。仅 `standard_relation='='` 且数值有效。
4. **日期（冻结）**：activity 的 `document_year > 2017`（≥2018）计为"快照后新增"。时间窗分层同 §5（2018-2020 / 2021-2026）。
5. **证据对定义**：(compound, gene) 存在 ≥1 条满足档位与日期的 activity。
6. 附加敏感性（可选报告）：organism=Homo sapiens 限定 vs 全物种。

## 7. CpD（palliates）— 冻结处置

**声明不可外部验证**。理由：试验注册不区分 treat/palliate 意图，与 CtD 合并将引入口径混杂并稀释 CtD 信号。论文 Limitations 如实声明（Master Prompt 给出的两选项中取更干净者）。

## 8. 统计检验（冻结）

对每条决策边（CtD×2 档时间窗、CbG×2 效力档×2 时间窗）：
1. **主检验**：在对齐后候选空间上 2×2 表 [KEEP/WITHHOLD] × [证据命中/未命中] 的 **Fisher 精确检验（单侧，KEEP 命中率更高）**。
2. **效应量**：lift = P(命中|KEEP)/P(命中|WITHHOLD)，两侧 **Wilson 95% CI**。
3. **随机对照**：等大随机集（从对齐后 absent 空间抽 |KEEP| 个，seed 1000-1009 共 **10 次**）命中率的均值±sd 与 KEEP 命中率的相对位置。
4. **多重性声明**：主结果=全局 τ 在两决策边的主时间窗（CbG Tier-1 全窗、CtD 全窗）；其余为敏感性，按探索性报告（不做事后校正，但全文列明检验总数）。
5. **可行性检查点**：对齐后证据命中对总数 <50（该边×档位）→ 报告功效受限（CI 宽），不推断。
6. **绝对规模上下文**：同时报告 KEEP 集大小、命中率绝对值（决策含义：每百条保留预测含多少条后续真实证据）。

## 9. 产物（冻结文件名）

```
external_validation/
├── protocol_frozen.md              ← 本文档
├── entity_alignment.md             ← 对齐率与丢弃明细（CtD 病/药两向、CbG 化合物/基因两向）
├── fetch_ctgov.py / fetch_chembl.py
├── export_candidates.py            ← 冻结阈值→全量候选 KEEP/WITHHOLD 导出（含实体标签）
├── run_enrichment.py               ← 富集检验
└── cache/                          ← API 原始响应（复现审计用）
results/external_validation_ctd.json
results/external_validation_cbg.json
analysis/external_validation_tables.md + fig_enrichment.{pdf,png}（Phase 2 出图）
```

## 10. 联网验证记录（W4/W5，2026-08-22）

| 项 | 结论 | 证据 |
|---|---|---|
| CTgov `filter.sasDateRangeStart` | ❌ 无效参数（API 报 unknown parameter） | curl 实测 |
| CTgov `filter.advanced=AREA[StartDate]RANGE[...]` | ✅ 生效 | curl 实测（返回近期研究） |
| CTgov `fields` 组合 | ⚠️ 部分路径触发空响应 | curl 实测 → 协议改取全量 |
| ChEMBL `activity_year__gte` | ❌ **静默无效**（不过滤） | order_by 实测：首条 document_year=2004 |
| ChEMBL `document_year` 字段 | ✅ 响应内嵌，改客户端过滤 | 实测响应含该字段 |
| ChEMBL inchikey / target_components(UniProt) | ✅ 过滤器与组装件可用 | 实测 |
| W4 先例 | Zhang 2021 JBI 即时间切片评估先例（"train prior, test on later trials"）；Bang 2023 Nat Commun / Kißig 2021 试验恢复验证 | 检索确认 |

## 11. 复现参数汇总（冻结）

模型=hetionet_core_RotatE.pt｜split seed=42（build_core_dataset(42)，与主审计同一 split）｜index 种子流 rng_ev=default_rng(42)（与主审计采样流分离）｜K=500｜τ 主=0.00399202｜时间窗 2017-01-01 与 2018-01-01 界（CTgov/ChEMBL 各按 §5/§6）｜效力档 10/100 nM｜随机对照 10×seed1000-1009｜Fisher 单侧。

---
name: deep-search
description: >
  深度搜索协议:多轮闭环(检索式→搜→读全文→抽主张→找缺口→改写→再搜),
  预算用尽才综合,每条结论回链原文;本地文献库优先,web 补缺。
---

# deep-search —— 深度搜索协议

"搜一下"是**一轮**;深度搜索是**一个闭环**,直到预算用尽才收口。产出不是
链接列表,是一份**每条结论都回链到你真读过的原文**的综合。

## 闭环(默认5轮,按任务调)

每一轮四步,缺一不可:

1. **检索**(本地优先):先在本地文献库找(见下"本地优先"),缺口才上网;
2. **读全文** —— 不是 snippet!PDF/文档用 `read`(documents 读取器会自己转),
   网页交给引擎或 browser-use 的页面快照;读不动的如实标注"仅读摘要";
3. **抽主张**:这一轮确认了什么?推翻了什么?**还缺什么**(→缺口);
4. **改写检索式**:针对缺口换关键词/换同义词/加限定(年份、方法、作者)→ 下一轮。

预算耗尽或缺口清零 → 进入综合。

## 两条检索通道

### 通道 A(主):本机引擎 `gpt-researcher`(已装0.15.1)

它自己会 搜→读→综合→带 visited_urls 引用,一次调用=一轮里最重的那步:

```bash
python - <<'EOF'
import asyncio, json
from gpt_researcher import GPTResearcher

async def main():
    r = GPTResearcher("你的检索式", report_type="research_report",
                      retriever="duckduckgo")   # ddgs 免 key;也可换其它
    await r.conduct_research()
    print("=== REPORT ===")
    print(await r.get_report())
    print("=== SOURCES ===")
    print(json.dumps(sorted(r.visited_urls), ensure_ascii=False, indent=1))

asyncio.run(main())
EOF
```

- `retriever="duckduckgo"` = **免 key**(本机已装 ddgs);有 key 的服务商见其文档;
- 本地文件夹参与:`GPTResearcher(query, report_type="…", doc_path="/path/to/library")`;
- 输出的 `visited_urls` **就是回链素材**,交给 `cite-sources` 的三查。

### 通道 B(兜底):我们的原生件

引擎装不上/跑挂 → 降级:`search_web`(已配 key)拿候选 → browser-use 快照
或 `read` 读全文 → 自己按上面四步推进。**兜底也要走完整闭环,不许退化成
"搜完就答"。**

## 本地优先(你自己的文献库才是第一检索源)

1. `bash` 里 `rg -i "关键词" <库目录> --glob '*.md' --glob '*.txt'`(已有抽取文本最快);
2. 没文本的 PDF → `read` 直接读(引擎自动转);
3. 命中 → 记 (文件, 页/节) 作为回链;**本地不够才上网**。

## 综合(收口)

- 每条结论 = 主张 + **回链**(本地=文件与页;web=URL)+ 访问深度标注
  (全文/摘要/转述)—— 规则照 `cite-sources`;
- 明确写清**没找到什么**(负结果也是结果);
- 分歧并列呈现,不硬凑一致;
- 最后一步可选:把综合稿丢给 `draft-critique` 的独立批评者过一遍
  (检查回链是否真支撑主张)。

## 纪律

- **预算意识**:开跑前告诉用户轮次与预计开销(每轮 = 若干次搜索+读取);
- **不许编造回链**:读不到全文的来源只能以"摘要级"身份出现;
- 中断恢复:每轮结束把"已确认/缺口/下轮检索式"写进当前回复 —— 丢进度
  比慢更糟。

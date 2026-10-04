# judge —— 判官(判断模型)

给对话接上一个**判断模型**:材料进、**带类型的问题**出,回来的是
`0/1 选择、分数、是非概率` 和一个**置信度** —— 不是散文。

背后是 [TypeSafe AI 的 Jev](https://www.jevai.org)(System One 模型,2026-09-15
限量发布),用法与设计依据见论文 [JEV-as-a-Judge: Accept When Confident,
Escalate When Unsure](https://arxiv.org/html/2609.26550v1)。

## 它为什么值得占一个工具位

- **快**:端到端 70~500 ms(官方数字),一次调用按 70–500 毫秒计 —— "决策前
  顺手问一句"的成本低到可以常态化;
- **便宜**:官方定价约 $0.04–0.084 / 百万输入词元,**输出免费**(比前沿 LLM
  低两个数量级);
- **结构化**:答案直接是软件能读的 JSON,不产生需要解析的自然语言。

## 契约(工具就是这个形状)

```jsonc
// 请求
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <key>
{ "model": "jev-latest",
  "state": "要判断的材料(文本或对象)",
  "questions": {
    "is_bug": { "type": "noul", "instructions": "是缺陷吗?",
                "criteria": { "true": "描述了异常行为", "false": "不是" } },
    "which":  { "type": "choice", "instructions": "选哪个",
                "choices": ["方案A", "方案B"] },
    "fit":    { "type": "score", "instructions": "完成度", "range": [0, 100] }
  } }

// 响应(节选,论文示例)
{ "answers": { "verdict": { "type": "choice", "choice": "supported",
                            "confidence": 1.0,
                            "probabilities": { "contradicted": 0.0,
                                               "unknown": 0.0,
                                               "supported": 1.0 } } },
  "usage": { "input_tokens": 416, "output_tokens": 42 } }
```

三种题型:**choice**(选项)、**score**(分数)、**noul**(是非 + 判据)。
回答里的 `confidence` 是**从概率分布汇总的统计量**,不是模型"自称有信心"。

## 配置(三个入口,等价)

| 入口 | 怎么写 |
|---|---|
| **桌面面板** | 设置里"判官"一节(接口地址 / 密钥 / 模型);**密钥永不回传面板**,留空即保留 |
| 命令行 | `ngu config set --judge-endpoint https://api.typesafe.ai --judge-key <key> --judge-model jev-latest` |
| 环境变量 | `NGU_JUDGE_API_KEY`(必填)、`NGU_JUDGE_ENDPOINT`、`NGU_JUDGE_MODEL` |

**工具出现规则和搜索一样**:`judge` 包装了 + 密钥配了 → 工具在表里;缺任何一个 →
工具不存在(一个注定 401 的工具,模型每试一次就白花一轮)。

## 红线:建议者,不是闸门

判官的结论**永远是上下文里的一条数据**:模型权衡、用户拍板,系统**不因为**
它说 `false` 拦任何动作,也**不因为**它说 `true` 免掉任何检查。低置信度的
正确用法是**补信息、换问法、或者问用户**,不是硬闯 —— 这与本内核"不设关卡、
如实报告"的总原则一致,soul 里也是这么要求模型的。

## 错误语义(工具报错时说什么)

| 状态 | 含义 | 工具返回 |
|---|---|---|
| 401 | 密钥缺失/无效 | 直接给出三个配置入口 |
| 422 | 请求不符合契约 | 附上响应体,指明是**契约问题**,别重试 |
| 429 / 529 | 限流 / 过载 | 建议退避重试,而不是循环打 |
| 其他 | 原样带状态码与响应片段 | — |

## 卸载

```bash
ngu uninstall judge        # 卸载:之后的对话不再有判官,文件保留
ngu uninstall judge --delete
```

**卸载 = 工具消失**(和 search 同一条语义:能力跟包走);你配的 `judge` 设置
**不删**,装回来即恢复。界面上"判官"一节也随之移除 —— 配一个没人能调用的
工具的设置,等于撒谎。

## 许可

包本身 MIT。**Jev 模型是 TypeSafe AI 的专有服务**,使用它需要向对方申请
API 访问(早期访问制);本包只实现调用契约,不包含、也不分发该模型。

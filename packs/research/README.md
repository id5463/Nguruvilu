# research —— 科研交付包

三个彼此独立又互相咬合的纪律,让"产出"达到可复核的标准:

| 内容 | 文件 | 作用 |
|---|---|---|
| 人设(standing) | `soul.md` | 三条硬规矩:每条主张带来源、写作与批评分开(actor-critic)、生成→批评→排名 |
| 技能 | `skills/cite-sources` | 找源、核对三查、诚实标注、引用格式 |
| 技能 | `skills/draft-critique` | 用 `delegate` 派独立批评者;同尺排名出比较表 |
| 技能 | `skills/publish-figures` | 参数化脚本生成、矢量输出、图自明、可反复修订 |

人设**按包拼接生效**(所有已装包的 `soul.md` 合成一份替换):装上它,中文人设
与科研纪律**同时在**;卸载它,科研纪律从下一次拼接里退出,历史记录原样保留。

## 借鉴来源(功能级,文字与实现全新)

- **ClawsGO Science**(clawsgo.cn):数小时持续的目标执行、**每条结论带来源**的
  交付、可反复修订的出版级图表 —— 我们取"带引用交付"与"修订循环"两条;
- **Anthropic Claude Science**:actor-critic 双代理(一个写、一个独立审)——
  取"批评者要独立上下文"这一条,执行用现成的 `delegate`;
- **Google Co-Scientist**:生成→批评→排名 —— 取"同一把尺子先定标准再看答案"。

以上均为**观察对方功能后的从零重写**:本包不含任何上游的代码、文案或提示词。

## 依赖

- `delegate`(来自 `starter`,已预装)—— 派批评者、扇出候选;
- 模型照常走 `NGU_API_KEY` / 面板配置;**不需要额外的密钥**。

## 安装与卸载

```bash
ngu pack packs/research --out packs/research/research-1.0.0.dshpack
ngu install packs/research/research-1.0.0.dshpack   # 或:ngu install github:...
ngu uninstall research           # 卸载:文件留着,装配不再加载
ngu uninstall research --delete  # 删除:连文件一起清掉
```

装载后技能进入目录(会话里用 `skill` 工具按 id 取),人设立即参与拼接;
按对话生效,已载入的对话在回合结束时换装,不重启。

## 许可

本包 MIT。三方产品的名字只出现在"借鉴来源"里,用于说明思路出处。

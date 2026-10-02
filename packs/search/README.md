# search —— 联网搜索

给对话接上 `search_web` 工具:按需搜索网页,把结果交给模型。

## 它怎么工作

这个包带两样东西:一个 `search.json`,和**一行插件声明**:

```json
{ "plugins": [ { "id": "search", "source": "builtin:search", "license": "MIT" } ] }
```

声明意味着:**`search_web` 工具属于这个包** —— 内核只提供代码,是这个包让对话里出现它。
没装这个包,工具就不会出现,哪怕设置里配了 key。

`search.json` 只带默认值:

```json
{
  "provider": "tavily",
  "apiKeyEnv": "TAVILY_API_KEY",
  "maxResults": 5
}
```

- **钥匙从不进包**。包里只写环境变量的**名字**;变量没设,工具就不注册 —— 模型看不到
  一个注定失败的工具,也就不会白花一轮去试它。
- 支持三家方言:**tavily**(POST,key 在请求体)、**brave**(`X-Subscription-Token`)、
  **exa**(`Authorization: Bearer`)。换家只需改设置,不用换包。
- **你配过的优先**:这个包随内核预装,所以它只补空缺,不会盖过你已经写好的设置。

## 拿到 key 之后

以 tavily 为例(https://tavily.com 注册即有):

```bash
# PowerShell
$env:TAVILY_API_KEY = "tvly-..."
ngu --new-session -q -p "现在能搜索吗?试着搜一下今天的新闻"

# 或者写进设置(会覆盖包里的 provider)
ngu config set --search-provider brave --search-api-key BRAVE_KEY
```

设置里也可以改端点(走代理 / 镜像):`--search-endpoint https://...`。

## 没有 key 时

什么都不会发生:`search_web` 不出现在工具表里。这是有意的 —— "不出现,模型就不知道
它存在",好过"出现了,每次调用都失败"。

## 卸载

```bash
ngu uninstall search        # 卸载,文件留着
```

**卸载 = 工具消失**:`search_web` 随这个包走,包不在,对话里就没有它 —— 这是"能力属于
包"的语义。你自己配的 `search` 设置**不删**,重新装上(或用 `ngu pack` / Apply 加载)
就恢复;`ngu uninstall search --delete` 才会连文件一起删,那时默认值也不再补进设置。

## 许可

MIT。

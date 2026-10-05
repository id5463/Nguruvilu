# documents —— 文档阅读包(读不了就装包)

**一个包教会 `read` 读 PDF 和 Office**:装上它,模型对 `.pdf / .pptx /
.docx / .xlsx` 直接 `read`;不装,`read` 会**如实说"没有读者认领这个格式,
装个包"并列出已装读者** —— 格式能力永远来自包,内核只负责"问谁会读"。

## 两个读者、三级策略

| 格式 | 走法 |
|---|---|
| **PDF** | ① 本机 Python 引擎(PyMuPDF 优先 —— CJK 字体表在它手里;markitdown 兜底)→ 失败给指令卡。**不做原生瞎猜**:手写解析器对着 CID 字体会输出自信的乱码,那比报错更糟 |
| **Office**(pptx/docx/xlsx/odt/odp/ods) | ① 引擎(markitdown)→ ② **内置 zip+XML 提取**(零依赖,新机器什么都没装也能读 PPT/DOCX)→ ③ 指令卡 |

## 失败即指令(缺环境就自己装)

读不出来时,错误信息里**直接带着修机器的命令** —— 模型有 `bash`、没有闸门,
看到就执行,然后重试即通:

- 没 Python → `winget install Python.Python.3.13`(Windows)/ `sudo apt install -y python3` / `brew install python`
- 有 Python 没库 → `python -m pip install --user pymupdf markitdown`
- 引擎跑了但没文本 → 如实说:**扫描件需要 OCR,不在本读取器范围内**

自愈闭环,不需要任何人在内核里写解析器。

## 依赖与边界

- **本包零第三方依赖**(内置提取只用内核已有的 zip;引擎用的是**你机器上
  已装的** Python,没装会告诉你怎么装);
- 图片**不是**读者:图片按消息协议作为附件内嵌,和"解读文件内容"是两回事,
  归内核管;
- PDF 的原生提取**刻意不做**(见上);XLSX 内置只提共享字符串表,
  完整表格走引擎。

## 安装与卸载

```bash
ngu pack packs/documents --out packs/documents/documents-1.0.0.dshpack
ngu install packs/documents/documents-1.0.0.dshpack
ngu uninstall documents           # 卸载后 read 回到"文本+图片"的内核原样
ngu uninstall documents --delete
```

按对话生效;卸载即摘除读者(注册是效果,卸载即撤销,归属面板可见
`documents ← documents`)。

## 许可

本包 MIT。

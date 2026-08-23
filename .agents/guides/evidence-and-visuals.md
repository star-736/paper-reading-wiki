# 证据与视觉材料

## 证据等级

任何非平凡机制主张都要显式采用正确等级：

1. **原文确证（primary-source confirmed）**：已直接重读 `raw/` PDF，使用如 `§ Instantiate DSA Under MLA`、`Figure 7 caption`、`Table 1` 的读者定位符；这是唯一可无保留写在“已据原文核实”下的等级。
2. **外部佐证（externally corroborated）**：独立外部来源支持、但不在原文中。必须给链接并注明是佐证；作者博客对意图的解释不能升级为等级 1。
3. **推断 / 本页原创综合（inference / original synthesis）**：来源未直接陈述的推理或框架。显式标注；机制猜测放入 `## 待追问`。

重读一手材料优先于相信既有摘要。不要把论文结论与自己的机制解释混在“已核实”陈述中；过度断言应降级，不应用弱佐证强撑。

## 图文化

页面是图文交错的。凡论文图或表承载读者无法仅凭文字重构的机制、架构或主要结果，都嵌入原图，不只写 `Figure N`。

- 图像放 `wiki/assets/<source-slug>/<figure-slug>.png`，英文 kebab-case，进入版本控制，且至少被一个页面引用。
- 用 PyMuPDF 以约 300 DPI 裁剪渲染：`page.get_pixmap(matrix=fitz.Matrix(300/72,300/72), clip=rect)`。根据 caption 和文本块确定边界，再用 `page.get_textbox(clip)` 检查图内标签完整、未混入 caption、相邻正文或页脚。不要盲信 `page.get_drawings()` bbox。
- 顶部图若串入前页脚注（例如 `Project Page:`、上标、分隔线），收紧 `clip.y0` 后重渲染。
- 纯文本或少量无意义线条的表改写成 Markdown 表；只有布局本身承载含义时才截图。
- 图片 alt text 是完整的读者向图注，随后给原论文 caption 的 blockquote 与定位符。
- `raw/` 图是原文确证；VLM 只能辅助阅读，不能升级图中未展示的机制结论。

## 何时使用 VLM

默认仅用 PyMuPDF 文本提取与 LLM 阅读。只有下列情形可调用 VLM：

| 触发条件 | 使用目的 |
| --- | --- |
| 单页有 ≥3 张表、跨页或合并单元格复杂表 | 还原结构化表格草稿，再人工核验 |
| ≥5 行 LaTeX 或复杂矩阵 / 张量公式 | 视觉确认关键符号、上下标与括号 |
| 架构图与相邻文字联合才有意义 | 辅助辨认模块与设计动机 |
| >10 张图或密集多子图 | 辅助分类 architecture / ablation / data / qualitative |
| 图注与正文矛盾 | 标出冲突，回到原文判断 |
| ≥5 个模块、多色数据流的复杂工作流图 | 辅助辨认边界与流向，再由正文核验 |

VLM 输出永远只是阅读辅助，不是证据等级；使用记录可写入日志，不进入读者正文。普通纯正文页或不满足以上条件的图不调用 VLM。

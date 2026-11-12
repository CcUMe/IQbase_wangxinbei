# C01 技术路线汇报材料

- `C01_电磁模型技术路线汇报.pptx`：20 页，16:9，可编辑正文与架构/流程图；各页含讲者备注。
- `C01_电磁模型技术路线汇报_预览.pdf`：同版面预览，便于快速查看与分享。
- `C01_汇报PPT_总览.png`：全部幻灯片缩略图。
- `C01_逐页讲稿.md`：逐页讲解、来源与常见问答。
- `build_presentation.py`：生成源码。真实输入示例和单页预览位于 `assets/`、`previews/`。

面向项目负责人，8–10 分钟；第 1–11 页为主讲，第 12–20 页为备查。PPT 字体指定微软雅黑，预览使用 Noto Sans CJK SC。PDF 是同源版面渲染预览，PPT 中的主体文字与图形保持可编辑。

内容来自当前仓库、全量数据审计、主配置参数统计、流程初测及已提供的研究材料。参数量与共享比例为小子集三步主配置实例，未呈现为正式训练精度。未改动原始数据、模型代码或权重。

复现依赖位于 `/tmp/c01_presentation_deps`，执行 `python docs/presentations/build_presentation.py`。若临时目录已清理，需先在隔离环境安装 python-pptx、Pillow、reportlab、matplotlib，并保证中文字体可用。

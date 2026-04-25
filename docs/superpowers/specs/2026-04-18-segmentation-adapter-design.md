# 第二阶段分割适配层设计

## 目标

在现有点击选点与 ROI 提取链路之上，新增一条可替换的分割链路：

- 浏览器点击点云
- 后端完成选点与 ROI 生成
- 分割适配层基于 ROI 和提示点输出最终点集 mask
- 前端同时显示 ROI 候选区域和最终提取结果

本阶段不直接集成真实 Point-SAM 权重，而是先搭建一层稳定的分割接口，并用启发式分割器跑通完整产品链路。后续只需要替换适配层实现即可接入真实模型。

## 设计结论

采用“可替换分割适配层 + 默认启发式分割器 + 新分割 API + 前端双层高亮”的方案。

原因：

- 当前仓库已经完成 `click -> pick -> roi`，但还没有模型依赖管理和推理适配边界。
- 直接把真实 Point-SAM 塞进现有服务会让点击链路、模型依赖和前端结果渲染同时耦合。
- 先固化接口，后续换成真实 Point-SAM 只需要替换分割器实现，不需要重写 API 和 UI。

## 架构

### 后端

新增 [point_selection/segmenter.py](/Users/gengchen/Desktop/3d/point_selection/segmenter.py) 作为分割适配层，提供：

- `SegmentConfig`
- `SegmentResult`
- `HeuristicRegionSegmenter`

`DemoService` 新增内部公共链路：

- `_resolve_pick_and_roi(payload)`：解析点击、相机和 ROI 参数，返回 `PointPickResult + ROIResult`
- `segment_roi(payload)`：在 `_resolve_pick_and_roi` 的基础上调用分割器，返回 `pick + roi + mask`
- `try_segment_roi(payload)`：捕获未命中场景，返回业务错误 JSON

新增 API：

- `POST /api/segment-roi`

返回结构：

```json
{
  "matched": true,
  "pick": { "...": "..." },
  "roi": { "...": "..." },
  "mask": {
    "seed_point_id": 110,
    "point_ids": [109, 110, 111],
    "point_count": 3,
    "confidence": 0.91,
    "method": "heuristic_region_growing"
  }
}
```

### 分割器实现

本阶段默认分割器为启发式区域生长：

- 使用点击点作为 seed
- 在 ROI 内基于空间邻接关系做 BFS/region growing
- 优先连接颜色相近且距离足够近的点
- 只输出与 seed 连通的那一簇

这不是最终模型效果，而是“可替换模型接口”的占位实现。后续真实 Point-SAM 适配器只需要遵守相同输入输出协议。

### 前端

Three.js 视图新增一层 `mask` 高亮：

- 基础点云：原始颜色
- ROI：暖色高亮
- 最终 mask：冷绿色高亮
- 选中点：亮蓝色标记

右侧状态面板保留紧凑布局，但调整指标：

- 继续显示场景、选中点和 ROI 信息
- 新增 `Mask 点数`
- 新增 `Mask 置信度`

## 数据流

1. 前端加载点云场景
2. 用户点击点云
3. 前端发送 `screen_x / screen_y + camera + pick + roi` 到 `POST /api/segment-roi`
4. 后端完成 pick 与 ROI 计算
5. 分割适配层基于 ROI 输出 mask
6. 前端同时渲染 ROI 和 mask，并刷新状态面板

## 范围

### 本轮包含

- 分割适配层接口
- 默认启发式分割器
- `POST /api/segment-roi`
- 前端 mask 图层
- 右侧面板显示 mask 结果
- 测试覆盖适配层和服务链路

### 本轮不包含

- 真实 Point-SAM 权重和推理环境
- 多提示点交互
- 负点提示
- 专门的门窗几何后处理
- 模型下载、显存管理和批处理优化

## 错误处理

- 点击未命中时，`segment-roi` 返回 `{ matched: false, message: ... }`
- 分割器异常时，返回 400/500 风格错误 JSON
- 前端保留当前场景，不清空点云，只清空结果层并显示错误提示

## 测试策略

- 单元测试：
  - 启发式分割器能返回包含 seed 的 mask
  - `DemoService.segment_roi()` 返回 `pick + roi + mask`
  - 未命中时 `try_segment_roi()` 返回业务 miss 载荷
- 冒烟测试：
  - 页面包含 mask 结果字段
- 浏览器验证：
  - 点击后 ROI 和 mask 同时可见
  - 右侧面板显示 mask 点数和置信度

## 交付结果

- 一个可替换的分割适配层
- 一个能输出最终点集 mask 的后端 API
- 一个在浏览器里可同时显示 ROI 和 mask 的第二阶段演示版

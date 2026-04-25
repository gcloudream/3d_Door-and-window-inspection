# Three.js 点云演示版 UI 设计

## 目标

构建一个本地 Web 演示界面，加载样例点云后提供以下能力：

- 在浏览器中显示点云场景
- 支持轨道控制查看场景
- 点击点云中的一个点
- 调用现有 Python 选点与 ROI 核心逻辑
- 高亮被选中的点与 ROI
- 在右侧面板展示调试信息与基础参数

## 设计结论

采用 `方案 A：左侧主视图 + 右侧信息栏` 的布局，以及 `Three.js 前端 + Python API` 的分层实现。

原因：

- 现有 Python 核心逻辑已经完成 `screen/camera -> ray -> point -> roi` 链路中的大部分计算，适合复用。
- 浏览器端更适合承担渲染、轨道控制、用户点击和状态展示。
- 右侧信息栏便于调试当前演示版，也能直接扩展为后续 Point-SAM 推理结果区。

## 架构

### 前端

使用原生 ES modules 和 Three.js 构建一个轻量单页应用：

- 加载点云数据并生成 `THREE.BufferGeometry`
- 使用 `THREE.Points` 显示基础点云
- 使用额外的 `THREE.Points` 图层显示 ROI 高亮
- 使用单独的 `THREE.Points` 或极小 marker 显示选中点
- 使用 `OrbitControls` 负责交互查看
- 监听鼠标点击并采集：
  - `screen_x`
  - `screen_y`
  - 相机位置
  - 相机朝向目标点
  - 相机 up 向量
  - 当前视口宽高
  - 相机内参近似值

### 后端

使用 Python 标准库 HTTP 服务提供两类能力：

- 静态文件服务：返回 HTML / CSS / JS / `node_modules/three`
- API 服务：
  - `GET /api/scene`：返回演示点云数据与包围盒元信息
  - `POST /api/pick-roi`：接收点击和参数，返回 `pick + roi`

后端内部直接复用：

- [point_selection/io.py](/Users/gengchen/Desktop/3d/point_selection/io.py)
- [point_selection/core.py](/Users/gengchen/Desktop/3d/point_selection/core.py)
- [point_selection/view_adapter.py](/Users/gengchen/Desktop/3d/point_selection/view_adapter.py)

## 数据流

1. 前端启动后请求 `GET /api/scene`
2. 后端读取默认样例点云并返回结构化点列表
3. 前端渲染点云并记录 `pointId -> buffer index`
4. 用户点击画布
5. 前端将点击坐标和当前相机参数发送到 `POST /api/pick-roi`
6. 后端调用现有选点和 ROI 核心逻辑返回结果
7. 前端根据返回的 `point_ids` 更新 ROI 图层和选中点图层
8. 右侧信息栏同步刷新

## 范围

### 本轮包含

- 单场景演示
- JSON / ASCII PLY 样例加载
- 点云基础显示
- 点击取点
- ROI 高亮
- 右侧参数面板
- 基础状态提示（加载中、未命中、错误）

### 本轮不包含

- 多场景管理
- 文件上传
- 二进制 PLY / PCD
- Point-SAM 推理
- 结果导出
- 大规模点云性能优化

## 错误处理

- 点云文件无法读取时，API 返回 500 和错误消息
- 点击未命中时，API 返回 404 风格的业务错误 JSON
- 前端在面板中显示错误状态，不清空当前可视结果
- 无效参数时，后端返回 400

## 测试策略

- Python 单元测试覆盖：
  - 场景序列化
  - pick-roi API 逻辑封装
  - 参数校验
- 手工验证覆盖：
  - 页面加载成功
  - 点云可旋转缩放
  - 点击后选中点与 ROI 高亮更新
  - 右侧参数与返回值一致

## 交付结果

- 一个可直接运行的本地 Web 演示页
- 一个可复用的 Python API 服务
- 保留当前 CLI 调试链路不受影响

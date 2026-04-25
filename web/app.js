import { SceneView } from "/web/scene-view.js";

const sceneName = document.querySelector("#scene-name");
const scenePointCount = document.querySelector("#scene-point-count");
const sceneBounds = document.querySelector("#scene-bounds");
const statusChip = document.querySelector("#status-chip");
const errorBox = document.querySelector("#error-box");
const hoverIndicator = document.querySelector("#hover-indicator");
const hoverPointId = document.querySelector("#hover-point-id");
const hoverPointXyz = document.querySelector("#hover-point-xyz");

const pickPointId = document.querySelector("#pick-point-id");
const pickXyz = document.querySelector("#pick-xyz");
const pickDistance = document.querySelector("#pick-distance");
const maskPointCount = document.querySelector("#mask-point-count");
const maskConfidence = document.querySelector("#mask-confidence");
const candidateLabel = document.querySelector("#candidate-label");
const candidateConfidence = document.querySelector("#candidate-confidence");
const candidateBoxShape = document.querySelector("#candidate-box-shape");
const candidateBoxAnchor = document.querySelector("#candidate-box-anchor");
const openingLabel = document.querySelector("#opening-label");
const openingConfidence = document.querySelector("#opening-confidence");
const openingReason = document.querySelector("#opening-reason");
const positivePointCount = document.querySelector("#positive-point-count");
const negativePointCount = document.querySelector("#negative-point-count");
const selectionAnchorPoint = document.querySelector("#selection-anchor-point");

const roiTitle = document.querySelector("#roi-title");
const roiPointCount = document.querySelector("#roi-point-count");
const roiRadiusResult = document.querySelector("#roi-radius-result");
const roiTruncated = document.querySelector("#roi-truncated");

const pointSizeScaleInput = document.querySelector("#point-size-scale");
const pointSizeScaleValue = document.querySelector("#point-size-scale-value");
const roiRadiusInput = document.querySelector("#roi-radius");
const roiRadiusValue = document.querySelector("#roi-radius-value");
const roiMinPointsInput = document.querySelector("#roi-min-points");
const roiMaxPointsInput = document.querySelector("#roi-max-points");
const maxDistanceInput = document.querySelector("#max-distance-to-ray");
const maxDistanceValue = document.querySelector("#max-distance-to-ray-value");
const augmentModeToggle = document.querySelector("#augment-mode-toggle");
const refineModeToggle = document.querySelector("#refine-mode-toggle");
const clearSelectionButton = document.querySelector("#clear-selection");
const sceneFileInput = document.querySelector("#scene-file");
const sceneDensityInput = document.querySelector("#scene-density");
const candidateFilterInput = document.querySelector("#candidate-filter");

const selectionTitle = document.querySelector("#selection-title");
const viewerElement = document.querySelector("#viewer");

const sceneView = new SceneView(viewerElement);
const interactionElement = sceneView.getInteractionElement();
const PICK_SCREEN_DISTANCE_PX = 28;
const DRAG_CLICK_SUPPRESSION_PX = 5;
let currentSelectionResult = null;
let interactionMode = "extract";
let pendingHoverPosition = null;
let hoverFrameId = null;
let lastHoverPosition = null;
let viewerPointerDownPosition = null;
let viewerPointerButton = null;
let isViewerDragging = false;
let suppressNextClick = false;
let activeViewerPointerId = null;
const DEFAULT_SCENE_DENSITY = 300000;
let currentSceneDensity = Number(sceneDensityInput.value || DEFAULT_SCENE_DENSITY);

let isRequestInFlight = false;
let isSceneLoading = false;

pointSizeScaleInput.addEventListener("input", () => {
  const scale = Number(pointSizeScaleInput.value);
  pointSizeScaleValue.textContent = `${scale.toFixed(1)}x`;
  sceneView.setPointSizeScale(scale);
});

roiRadiusInput.addEventListener("input", () => {
  roiRadiusValue.textContent = `${Number(roiRadiusInput.value).toFixed(1)} m`;
});

maxDistanceInput.addEventListener("input", () => {
  maxDistanceValue.textContent = `${Number(maxDistanceInput.value).toFixed(2)} m`;
  if (lastHoverPosition && !isSceneLoading && !isRequestInFlight) {
    scheduleHoverFeedback(lastHoverPosition);
  }
});

clearSelectionButton.addEventListener("click", () => {
  clearPendingHoverFeedback();
  sceneView.clearSelection();
  sceneView.clearHover();
  currentSelectionResult = null;
  interactionMode = "extract";
  clearSelectionState();
  renderHoverFeedback(null);
  updateRefineModeUI();
  setStatus("已清空高亮");
});

augmentModeToggle.addEventListener("click", () => {
  if (!currentSelectionResult) {
    setStatus("请先完成一次目标提取，再进入补充模式");
    return;
  }

  interactionMode = interactionMode === "augment" ? "extract" : "augment";
  hideError();
  updateRefineModeUI();
  setStatus(
    interactionMode === "augment"
      ? "补充模式已开启，点击同一门窗的遗漏区域即可扩展结果"
      : "已返回普通提取模式",
  );
});

refineModeToggle.addEventListener("click", () => {
  if (!currentSelectionResult) {
    setStatus("请先完成一次目标提取，再进入排除模式");
    return;
  }

  interactionMode = interactionMode === "exclude" ? "extract" : "exclude";
  hideError();
  updateRefineModeUI();
  setStatus(
    interactionMode === "exclude"
      ? "排除模式已开启，点击误选区域即可收紧结果"
      : "已返回普通提取模式",
  );
});

candidateFilterInput.addEventListener("change", () => {
  sceneView.setClassificationFilter(candidateFilterInput.value);
  if (!currentSelectionResult) {
    setStatus("已更新候选筛选");
    return;
  }
  if (isSelectionVisible(currentSelectionResult)) {
    setStatus(`筛选已更新 · ${formatCandidateLabel(currentSelectionResult.classification?.label)}`);
  } else {
    setStatus("当前候选已根据筛选隐藏");
  }
});

sceneDensityInput.addEventListener("change", async () => {
  if (isSceneLoading || isRequestInFlight) {
    sceneDensityInput.value = String(currentSceneDensity);
    return;
  }

  const nextDensity = getSelectedSceneDensity();
  const previousDensity = currentSceneDensity;
  setSceneLoadingState(true, `正在按 ${formatSceneDensityLabel(nextDensity)} 重新加载当前场景`);

  try {
    const reloadResponse = await fetch("/api/scene/reload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_points: nextDensity }),
    });
    const metaPayload = await reloadResponse.json();
    if (!reloadResponse.ok) {
      throw new Error(metaPayload.error || "场景重载失败");
    }
    const pointsResponse = await fetch("/api/scene/points");
    const pointsPayload = await pointsResponse.json();
    if (!pointsResponse.ok) {
      throw new Error(pointsPayload.error || "点云数据加载失败");
    }
    const payload = { ...metaPayload, points: pointsPayload.points };
    applyScenePayload(payload);
    setStatus(sceneReadyMessage(payload));
  } catch (error) {
    currentSceneDensity = previousDensity;
    sceneDensityInput.value = String(previousDensity);
    showError(error.message);
    setStatus("点云密度切换失败，请稍后重试");
  } finally {
    setSceneLoadingState(false);
  }
});

sceneFileInput.addEventListener("change", async () => {
  const [file] = sceneFileInput.files ?? [];
  if (!file || isSceneLoading) {
    return;
  }

  isSceneLoading = true;
  hideError();
  setStatus(`正在加载 ${file.name}`);

  try {
    const response = await fetch("/api/scene/load", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Scene-Filename": encodeURIComponent(file.name),
        "X-Scene-Max-Points": String(getSelectedSceneDensity()),
      },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "场景加载失败");
    }

    applyScenePayload(payload);
    setStatus(sceneReadyMessage(payload));
  } catch (error) {
    showError(error.message);
    setStatus("文件加载失败");
  } finally {
    isSceneLoading = false;
    sceneFileInput.value = "";
  }
});

interactionElement.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});

interactionElement.addEventListener("pointerdown", (event) => {
  if (typeof interactionElement.setPointerCapture === "function") {
    try {
      interactionElement.setPointerCapture(event.pointerId);
      activeViewerPointerId = event.pointerId;
    } catch (_error) {
      activeViewerPointerId = null;
    }
  }
  viewerPointerDownPosition = {
    clientX: event.clientX,
    clientY: event.clientY,
  };
  viewerPointerButton = event.button;
  isViewerDragging = false;
  viewerElement.classList.add("is-view-dragging");
  clearPendingHoverFeedback();
  sceneView.clearHover();
  renderHoverFeedback(null);
});

interactionElement.addEventListener("pointermove", (event) => {
  if (isSceneLoading) {
    return;
  }

  if (viewerPointerDownPosition) {
    const dragDistance = Math.hypot(
      event.clientX - viewerPointerDownPosition.clientX,
      event.clientY - viewerPointerDownPosition.clientY,
    );
    if (dragDistance >= DRAG_CLICK_SUPPRESSION_PX) {
      isViewerDragging = true;
      if (viewerPointerButton === 0) {
        suppressNextClick = true;
      }
    }
    clearPendingHoverFeedback();
    sceneView.clearHover();
    renderHoverFeedback(null);
    return;
  }

  const rect = interactionElement.getBoundingClientRect();
  const hoverPosition = {
    screenX: event.clientX - rect.left,
    screenY: event.clientY - rect.top,
  };
  scheduleHoverFeedback(hoverPosition);
});

interactionElement.addEventListener("pointerleave", () => {
  if (viewerPointerDownPosition) {
    return;
  }
  suppressNextClick = false;
  resetViewerDragState();
  clearPendingHoverFeedback();
  sceneView.clearHover();
  renderHoverFeedback(null);
});

interactionElement.addEventListener("pointerup", (event) => {
  if (isViewerDragging && viewerPointerButton === 0) {
    suppressNextClick = true;
    window.setTimeout(() => {
      suppressNextClick = false;
    }, 250);
  }
  releaseViewerPointerCapture(event.pointerId);
  resetViewerDragState();
});

interactionElement.addEventListener("pointercancel", (event) => {
  suppressNextClick = false;
  releaseViewerPointerCapture(event.pointerId);
  resetViewerDragState();
  clearPendingHoverFeedback();
});

interactionElement.addEventListener("click", async (event) => {
  if (suppressNextClick) {
    event.preventDefault();
    suppressNextClick = false;
    setStatus("已调整视角");
    return;
  }
  if (isRequestInFlight || isSceneLoading) {
    return;
  }

  isRequestInFlight = true;
  hideError();
  setStatus("正在计算 ROI 与分割结果");

  const rect = interactionElement.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;
  const previousSelection = currentSelectionResult;
  const hint = sceneView.findPointHint(localX, localY, {
    maxPixelDistance: PICK_SCREEN_DISTANCE_PX,
    maxDistanceToRay: Number(maxDistanceInput.value),
  });
  const isAugmentRequest = interactionMode === "augment" && Boolean(currentSelectionResult);
  const isExcludeRequest = interactionMode === "exclude" && Boolean(currentSelectionResult);
  const isRefineRequest = isAugmentRequest || isExcludeRequest;
  const selectionSeedPointId = isRefineRequest ? getSelectionSeedPointId(currentSelectionResult) : null;

  if (isRefineRequest && !hint) {
    showError(isAugmentRequest ? "补充模式下请点击点云中的目标区域" : "排除模式下请点击点云中的误选区域");
    setStatus(isAugmentRequest ? "正样本点击未命中，请重试" : "负样本点击未命中，请重试");
    isRequestInFlight = false;
    return;
  }

  const payload = {
    ...buildPickRequestPayload(
      localX,
      localY,
      {
        hintedPointId: isRefineRequest ? hint?.pointId ?? selectionSeedPointId : hint?.pointId ?? null,
        lockedPointId: hint?.pointId ?? null,
      },
    ),
    roi: {
      radius: Number(roiRadiusInput.value),
      min_points: Number(roiMinPointsInput.value),
      max_points: Number(roiMaxPointsInput.value),
    },
  };
  if (isRefineRequest) {
    payload.refine = {
      seed_point_id: selectionSeedPointId,
      positive_point_ids: isAugmentRequest
        ? buildNextPositivePointIds(currentSelectionResult, hint.pointId)
        : getPositivePointIds(currentSelectionResult),
      negative_point_ids: isExcludeRequest
        ? buildNextNegativePointIds(currentSelectionResult, hint.pointId)
        : getNegativePointIds(currentSelectionResult),
    };
    setStatus(isAugmentRequest ? "正在根据正样本扩展结果" : "正在根据负样本收紧结果");
  } else {
    setStatus("正在计算 ROI 与分割结果");
  }

  try {
    const response = await fetch("/api/segment-roi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "选点失败");
    }
    if (!result.matched) {
      if (!isRefineRequest) {
        sceneView.clearSelection();
        currentSelectionResult = null;
        interactionMode = "extract";
        clearSelectionState();
        updateRefineModeUI();
      }
      showError(result.message || "没有命中点云");
      setStatus(
        isAugmentRequest
          ? "正样本点击未命中，请重试"
          : isExcludeRequest
            ? "负样本点击未命中，请重试"
            : "点击未命中，请靠近点云后重试",
      );
      return;
    }
    sceneView.setSelection(result);
    sceneView.focusOnPointById(getSelectionSeedPointId(result));
    currentSelectionResult = result;
    renderSelection(result);
    updateRefineModeUI();
    if (isAugmentRequest) {
      const previousPointCount = Number(previousSelection?.mask?.point_count ?? 0);
      const addedCount = Math.max(0, result.mask.point_count - previousPointCount);
      if (addedCount > 0) {
        setStatus(`已补充 ${addedCount} 个点`);
      } else {
        setStatus("当前正样本没有扩展结果");
      }
    } else if (isExcludeRequest) {
      const removedCount = Number(result.refinement?.removed_point_count ?? 0);
      if (removedCount > 0) {
        setStatus(`已排除 ${removedCount} 个误选点`);
      } else {
        setStatus("当前负样本没有命中可剔除区域");
      }
    } else if (isSelectionVisible(result)) {
      setStatus(`已提取 ${result.mask.point_count} 个点 · ${formatCandidateLabel(result.classification?.label)}`);
    } else {
      setStatus(`已提取 ${result.mask.point_count} 个点，但已被筛选隐藏`);
    }
  } catch (error) {
    showError(error.message);
    setStatus("点击未命中或参数无效");
  } finally {
    isRequestInFlight = false;
  }
});

bootstrap().catch((error) => {
  showError(error.message);
  setStatus("场景加载失败");
});

async function bootstrap() {
  const metaResponse = await fetch("/api/scene/meta");
  const metaPayload = await metaResponse.json();
  if (!metaResponse.ok) {
    throw new Error(metaPayload.error || "场景加载失败");
  }

  setSceneLoadingState(true, `正在加载点云数据 · ${sceneReadyMessage(metaPayload)}`);

  const sceneResponse = await fetch("/api/scene");
  const scenePayload = await sceneResponse.json();
  if (!sceneResponse.ok) {
    throw new Error(scenePayload.error || "点云数据加载失败");
  }

  applyScenePayload(scenePayload);
  setSceneLoadingState(false);
  setStatus(sceneReadyMessage(scenePayload));
}

function applyScenePayload(payload) {
  currentSceneDensity = Number(payload.sampling?.max_points ?? getSelectedSceneDensity());
  sceneDensityInput.value = String(currentSceneDensity);
  sceneView.setPointCloud(payload);
  sceneView.setPointSizeScale(Number(pointSizeScaleInput.value));
  sceneView.setClassificationFilter(candidateFilterInput.value);
  clearPendingHoverFeedback();
  sceneView.clearHover();
  currentSelectionResult = null;
  interactionMode = "extract";
  renderScene(payload);
  clearSelectionState();
  renderHoverFeedback(null);
  updateRefineModeUI();
}

function renderScene(payload) {
  sceneName.textContent = payload.scene_name;
  scenePointCount.textContent = String(payload.point_count);
  sceneBounds.textContent = `${formatVector(payload.bounds.min)} -> ${formatVector(payload.bounds.max)}`;
  pointSizeScaleValue.textContent = `${Number(pointSizeScaleInput.value).toFixed(1)}x`;
}

function renderSelection(result) {
  const roiPoints = result.roi.points ?? [];
  const maskPoints = result.mask.points ?? [];
  const selectionSeedPointId = getSelectionSeedPointId(result);
  const anchorPoint = sceneView.getSelectionPointById(selectionSeedPointId);
  const anchorXyz = anchorPoint?.xyz ?? result.pick.xyz;
  const triggeredByRefinePoint = selectionSeedPointId !== result.pick.point_id;

  selectionTitle.textContent = `Point #${selectionSeedPointId} · ${formatCandidateLabel(result.classification?.label)}`;
  pickPointId.textContent = String(selectionSeedPointId);
  pickXyz.textContent = formatVector(anchorXyz);
  selectionAnchorPoint.textContent = String(selectionSeedPointId);
  pickDistance.textContent = triggeredByRefinePoint
    ? `点击点 #${result.pick.point_id} · ${result.pick.distance_to_ray.toFixed(4)} m`
    : `${result.pick.distance_to_ray.toFixed(4)} m`;
  maskPointCount.textContent = String(maskPoints.length || result.mask.point_count);
  maskConfidence.textContent = `${Number(result.mask.confidence).toFixed(2)}`;
  candidateLabel.textContent = formatCandidateLabel(result.classification?.label);
  candidateConfidence.textContent = result.classification
    ? `${Number(result.classification.confidence).toFixed(2)}`
    : "-";
  candidateBoxShape.textContent = formatCandidateBoxShape(result.candidate_box?.shape);
  candidateBoxAnchor.textContent = formatCandidateAnchor(result.candidate_box?.anchor_mode);
  openingLabel.textContent = formatCandidateLabel(result.opening_candidate?.label);
  openingConfidence.textContent = result.opening_candidate
    ? `${Number(result.opening_candidate.confidence).toFixed(2)}`
    : "-";
  openingReason.textContent = result.opening_candidate?.reason ?? "-";
  positivePointCount.textContent = String(getPositivePointIds(result).length);
  negativePointCount.textContent = String(getNegativePointIds(result).length);

  roiTitle.textContent = `${roiPoints.length || result.roi.point_ids.length} points`;
  roiPointCount.textContent = String(roiPoints.length || result.roi.point_ids.length);
  roiRadiusResult.textContent = `${Number(result.roi.radius).toFixed(2)} m`;
  roiTruncated.textContent = result.roi.truncated ? "Yes" : "No";
}

function clearSelectionState() {
  selectionTitle.textContent = "等待点击";
  pickPointId.textContent = "-";
  pickXyz.textContent = "-";
  pickDistance.textContent = "-";
  maskPointCount.textContent = "-";
  maskConfidence.textContent = "-";
  candidateLabel.textContent = "-";
  candidateConfidence.textContent = "-";
  candidateBoxShape.textContent = "-";
  candidateBoxAnchor.textContent = "-";
  openingLabel.textContent = "-";
  openingConfidence.textContent = "-";
  openingReason.textContent = "-";
  positivePointCount.textContent = "0";
  negativePointCount.textContent = "0";
  selectionAnchorPoint.textContent = "-";
  roiTitle.textContent = "未生成";
  roiPointCount.textContent = "-";
  roiRadiusResult.textContent = "-";
  roiTruncated.textContent = "-";
  hideError();
}

function flushHoverFeedback() {
  hoverFrameId = null;
  if (!pendingHoverPosition || isSceneLoading || isRequestInFlight) {
    return;
  }

  const hoverPosition = pendingHoverPosition;
  pendingHoverPosition = null;
  applyLocalHoverFeedback(hoverPosition);
}

function scheduleHoverFeedback(position) {
  pendingHoverPosition = position;
  lastHoverPosition = position;
  if (hoverFrameId !== null) {
    return;
  }
  hoverFrameId = window.requestAnimationFrame(flushHoverFeedback);
}

function applyLocalHoverFeedback(position) {
  const hoverInfo = sceneView.updateHoverFromScreen(
    position.screenX,
    position.screenY,
    {
      exactPixelMatchOnly: true,
      maxPixelDistance: PICK_SCREEN_DISTANCE_PX,
      maxDistanceToRay: Number(maxDistanceInput.value),
    },
  );
  renderHoverFeedback(hoverInfo);
}

function clearPendingHoverFeedback() {
  lastHoverPosition = null;
  pendingHoverPosition = null;
  if (hoverFrameId !== null) {
    window.cancelAnimationFrame(hoverFrameId);
    hoverFrameId = null;
  }
}

function resetViewerDragState() {
  viewerPointerDownPosition = null;
  viewerPointerButton = null;
  isViewerDragging = false;
  viewerElement.classList.remove("is-view-dragging");
}

function releaseViewerPointerCapture(pointerId = null) {
  if (activeViewerPointerId === null) {
    return;
  }
  if (pointerId !== null && pointerId !== activeViewerPointerId) {
    return;
  }
  if (typeof interactionElement.releasePointerCapture === "function") {
    try {
      interactionElement.releasePointerCapture(activeViewerPointerId);
    } catch (_error) {
      // Ignore release errors when capture was already cleared by the browser.
    }
  }
  activeViewerPointerId = null;
}

function buildPickRequestPayload(screenX, screenY, pickOptions = {}) {
  return {
    screen_x: screenX,
    screen_y: screenY,
    camera: sceneView.getCameraPayload(),
    pick: {
      max_distance_to_ray: Number(maxDistanceInput.value),
      max_screen_distance_px: PICK_SCREEN_DISTANCE_PX,
      hinted_point_id: pickOptions.hintedPointId ?? null,
      locked_point_id: pickOptions.lockedPointId ?? null,
    },
  };
}

function isSelectionVisible(result) {
  const filterValue = candidateFilterInput.value;
  const label = result?.classification?.label ?? "unknown";
  return filterValue === "all" || filterValue === label;
}

function formatVector(vector) {
  return vector.map((value) => Number(value).toFixed(2)).join(", ");
}

function setStatus(message) {
  statusChip.textContent = message;
}

function setSceneLoadingState(loading, message) {
  isSceneLoading = loading;
  sceneDensityInput.disabled = loading;
  sceneFileInput.disabled = loading;
  augmentModeToggle.disabled = loading;
  refineModeToggle.disabled = loading;
  clearSelectionButton.disabled = loading;
  if (loading) {
    viewerElement.classList.add("is-loading-scene");
    sceneView.setInteractionEnabled(false);
    setStatus(message ?? "正在加载场景");
  } else {
    viewerElement.classList.remove("is-loading-scene");
    sceneView.setInteractionEnabled(true);
  }
}

function renderHoverFeedback(hoverInfo) {
  if (!hoverInfo) {
    hoverPointId.textContent = "-";
    hoverPointXyz.textContent = "-";
    hoverIndicator.classList.add("hidden");
    return;
  }

  hoverPointId.textContent = `Point #${hoverInfo.pointId}`;
  hoverPointXyz.textContent = formatVector(hoverInfo.xyz);
  hoverIndicator.classList.remove("hidden");
}

function updateRefineModeUI() {
  const hasSelection = Boolean(currentSelectionResult);
  augmentModeToggle.disabled = !hasSelection;
  refineModeToggle.disabled = !hasSelection;
  augmentModeToggle.classList.toggle("is-active", interactionMode === "augment" && hasSelection);
  augmentModeToggle.classList.toggle("is-augment-active", interactionMode === "augment" && hasSelection);
  refineModeToggle.classList.toggle("is-active", interactionMode === "exclude" && hasSelection);
  refineModeToggle.classList.toggle("is-negative-active", interactionMode === "exclude" && hasSelection);
  augmentModeToggle.textContent = interactionMode === "augment" && hasSelection ? "退出补充模式" : "补充模式";
  refineModeToggle.textContent = interactionMode === "exclude" && hasSelection ? "退出排除模式" : "排除模式";
}

function buildNextPositivePointIds(result, pointId) {
  const positiveIds = new Set(getPositivePointIds(result));
  positiveIds.add(pointId);
  return Array.from(positiveIds);
}

function buildNextNegativePointIds(result, pointId) {
  const negativeIds = new Set(getNegativePointIds(result));
  negativeIds.add(pointId);
  return Array.from(negativeIds);
}

function getSelectionSeedPointId(result) {
  return result?.mask?.seed_point_id ?? result?.pick?.point_id ?? null;
}

function getPositivePointIds(result) {
  return result?.mask?.positive_point_ids ?? result?.refinement?.positive_point_ids ?? [];
}

function getNegativePointIds(result) {
  return result?.mask?.negative_point_ids ?? result?.refinement?.negative_point_ids ?? [];
}

function formatCandidateLabel(label) {
  if (label === "door") {
    return "门候选";
  }
  if (label === "window") {
    return "窗候选";
  }
  if (label === "unknown") {
    return "未识别";
  }
  return "-";
}

function formatCandidateBoxShape(shape) {
  if (shape === "wall_rect") {
    return "贴墙矩形";
  }
  if (shape === "aabb") {
    return "轴对齐框";
  }
  if (shape === "empty") {
    return "空结果";
  }
  return "-";
}

function formatCandidateAnchor(anchorMode) {
  if (anchorMode === "wall_plane") {
    return "墙面锚定";
  }
  if (anchorMode === "mask_face") {
    return "保留原面";
  }
  if (anchorMode === "bounds") {
    return "包围盒";
  }
  return "-";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function segmentationReadyMessage(segmentation, sceneName) {
  if (!segmentation) {
    return `已加载 ${sceneName}`;
  }
  if (segmentation.active_backend === "point_sam") {
    return `Point-SAM 已就绪：${sceneName}`;
  }
  if (segmentation.requested_backend === "point_sam" && segmentation.fallback_reason) {
    return `Point-SAM 未就绪，已回退启发式分割`;
  }
  return `启发式分割已就绪：${sceneName}`;
}

function sceneReadyMessage(payload) {
  const baseMessage = segmentationReadyMessage(payload.segmentation, payload.scene_name);
  const densityLabel = formatSceneDensityLabel(payload.sampling?.max_points ?? currentSceneDensity);
  if (payload.scene_name?.toLowerCase().endsWith(".pcd")) {
    return `${baseMessage} · ${densityLabel} · 已支持 PCD`;
  }
  return `${baseMessage} · ${densityLabel}`;
}

function getSelectedSceneDensity() {
  const parsed = Number(sceneDensityInput.value);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return DEFAULT_SCENE_DENSITY;
  }
  return Math.round(parsed);
}

function formatSceneDensityLabel(maxPoints) {
  const resolved = Number(maxPoints);
  if (!Number.isFinite(resolved) || resolved <= 0) {
    return "默认密度";
  }
  if (resolved <= 180000) {
    return "流畅档";
  }
  if (resolved <= 300000) {
    return "推荐档";
  }
  if (resolved <= 600000) {
    return "精细档";
  }
  return "高精度档";
}

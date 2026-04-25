import * as THREE from "https://unpkg.com/three@0.170.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.170.0/examples/jsm/controls/OrbitControls.js";
import { buildHintCandidateKey, computeSquareHitMetrics } from "/web/hit-testing.js";
import { encodePickIndexToRgb, findNearestPickHit } from "/web/pick-buffer.js";

export class SceneView {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(recommendedPixelRatio(0));
    this.renderer.sortObjects = false;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.46;
    this.controls.panSpeed = 0.92;
    this.controls.zoomSpeed = 0.96;
    this.controls.enablePan = true;
    this.controls.enableZoom = true;
    this.controls.enableRotate = true;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.minPolarAngle = 0.08;
    this.controls.maxPolarAngle = Math.PI - 0.08;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.roundPointTexture = createRoundPointTexture();

    this.baseGeometry = new THREE.BufferGeometry();
    this.baseMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.05,
      vertexColors: true,
      opacity: 0.82,
    });
    this.basePoints = new THREE.Points(this.baseGeometry, this.baseMaterial);

    this.highlightGeometry = new THREE.BufferGeometry();
    this.highlightMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.11,
      color: "#ffb35c",
    });
    this.highlightPoints = new THREE.Points(this.highlightGeometry, this.highlightMaterial);

    this.maskGeometry = new THREE.BufferGeometry();
    this.maskMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.14,
      color: "#86f0ce",
      opacity: 0.96,
    });
    this.maskPoints = new THREE.Points(this.maskGeometry, this.maskMaterial);

    this.selectedGeometry = new THREE.BufferGeometry();
    this.selectedMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.2,
      color: "#8cc4ff",
    });
    this.selectedPoint = new THREE.Points(this.selectedGeometry, this.selectedMaterial);

    this.hoverGeometry = new THREE.BufferGeometry();
    this.hoverMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.16,
      color: "#fff3d6",
      opacity: 0.98,
    });
    this.hoverPoint = new THREE.Points(this.hoverGeometry, this.hoverMaterial);

    this.negativeGeometry = new THREE.BufferGeometry();
    this.negativeMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.18,
      color: "#ff7c70",
      opacity: 0.96,
    });
    this.negativePoints = new THREE.Points(this.negativeGeometry, this.negativeMaterial);

    this.positiveGeometry = new THREE.BufferGeometry();
    this.positiveMaterial = createRoundPointsMaterial(this.roundPointTexture, {
      size: 0.18,
      color: "#ffd866",
      opacity: 0.96,
    });
    this.positivePoints = new THREE.Points(this.positiveGeometry, this.positiveMaterial);

    this.candidateGeometry = new THREE.BufferGeometry();
    this.candidateMaterial = new THREE.LineBasicMaterial({
      color: "#86f0ce",
      transparent: true,
      opacity: 0.95,
    });
    this.candidateOutline = new THREE.LineSegments(this.candidateGeometry, this.candidateMaterial);
    this.candidateOutline.visible = false;

    this.pickScene = new THREE.Scene();
    this.pickGeometry = new THREE.BufferGeometry();
    this.pickMaterial = createPickMaterial();
    this.pickPoints = new THREE.Points(this.pickGeometry, this.pickMaterial);
    this.pickPoints.frustumCulled = false;
    this.pickScene.add(this.pickPoints);

    this.pointLookup = new Map();
    this.selectionPointLookup = new Map();
    this.pickIndexToPointId = [];
    this.points = [];
    this.projectedPointBuckets = new Map();
    this.projectedPointBucketSizePx = 28;
    this.projectedPointBucketsSignature = "";
    this.projectedPointMinCameraDepth = Number.POSITIVE_INFINITY;
    this.pickRenderTarget = null;
    this.pickReadBuffer = new Uint8Array(0);
    this.pickBufferNeedsRender = true;
    this.drawingBufferSize = new THREE.Vector2();
    this.projectedVector = new THREE.Vector3();
    this.raySampleVector = new THREE.Vector3();
    this.rayDirectionVector = new THREE.Vector3();
    this.relativeVector = new THREE.Vector3();
    this.cameraSpaceVector = new THREE.Vector3();
    this.clearColorValue = new THREE.Color();
    this.currentSelection = null;
    this.currentHoverPointId = null;
    this.classificationFilter = "all";
    this.basePointSize = 0.05;
    this.pointSizeScale = 1.0;

    this.scene.add(
      this.basePoints,
      this.highlightPoints,
      this.maskPoints,
      this.selectedPoint,
      this.hoverPoint,
      this.negativePoints,
      this.positivePoints,
      this.candidateOutline,
    );
    this.scene.add(new THREE.AmbientLight("#ffffff", 0.55));

    const keyLight = new THREE.DirectionalLight("#ffe9c8", 1.2);
    keyLight.position.set(4, 6, 7);
    this.scene.add(keyLight);

    this.renderer.domElement.className = "scene-canvas";
    this.container.appendChild(this.renderer.domElement);

    this.handleResize = this.handleResize.bind(this);
    this.handleCameraChange = this.handleCameraChange.bind(this);
    this.controls.addEventListener("change", this.handleCameraChange);
    window.addEventListener("resize", this.handleResize);
    this.handleResize();
    this.animate();
  }

  setPointCloud(scenePayload) {
    this.renderer.setPixelRatio(recommendedPixelRatio(scenePayload.point_count));
    this.handleResize();
    this.points = scenePayload.points;
    this.pointLookup = new Map(scenePayload.points.map((point) => [point.point_id, point]));
    this.selectionPointLookup = new Map();
    this.pickIndexToPointId = scenePayload.points.map((point) => point.point_id);
    this.invalidateProjectedPointBuckets();
    this.invalidatePickBuffer();

    const positions = [];
    const colors = [];
    const pickColors = [];

    for (const [index, point] of scenePayload.points.entries()) {
      positions.push(...point.xyz);
      colors.push(point.rgb[0] / 255, point.rgb[1] / 255, point.rgb[2] / 255);
      const [pickRed, pickGreen, pickBlue] = encodePickIndexToRgb(index + 1);
      pickColors.push(pickRed / 255, pickGreen / 255, pickBlue / 255);
    }

    const positionArray = new Float32Array(positions);
    const colorArray = new Float32Array(colors);
    const pickColorArray = new Float32Array(pickColors);

    this.baseGeometry.dispose();
    this.baseGeometry = new THREE.BufferGeometry();
    this.baseGeometry.setAttribute("position", new THREE.BufferAttribute(positionArray, 3));
    this.baseGeometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
    this.basePoints.geometry = this.baseGeometry;

    this.pickGeometry.dispose();
    this.pickGeometry = new THREE.BufferGeometry();
    this.pickGeometry.setAttribute("position", new THREE.BufferAttribute(positionArray.slice(), 3));
    this.pickGeometry.setAttribute("color", new THREE.BufferAttribute(pickColorArray, 3));
    this.pickPoints.geometry = this.pickGeometry;
    this.basePointSize = recommendedPointSize(scenePayload);
    this.applyPointSizes();

    this.frameBounds(scenePayload.bounds);
    this.clearHover();
    this.clearSelection();
  }

  setPointSizeScale(scale) {
    const nextScale = Number.isFinite(scale) ? Math.max(0.2, scale) : 1.0;
    this.pointSizeScale = nextScale;
    this.applyPointSizes();
  }

  frameBounds(bounds) {
    const min = new THREE.Vector3(...bounds.min);
    const max = new THREE.Vector3(...bounds.max);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const size = max.clone().sub(min);
    const radius = Math.max(size.length() * 0.5, 0.6);
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
    const fitDistance = Math.max(
      radius / Math.tan(verticalFov / 2),
      radius / Math.tan(horizontalFov / 2),
    );
    const viewDirection = new THREE.Vector3(0.18, 0.34, 1).normalize();
    const distance = fitDistance * 1.15;

    this.controls.target.copy(center);
    this.controls.minDistance = Math.max(radius * 0.18, 0.12);
    this.controls.maxDistance = Math.max(distance * 4.5, this.controls.minDistance + 1);
    this.camera.position.copy(center.clone().addScaledVector(viewDirection, distance));
    this.camera.near = 0.01;
    this.camera.far = distance * 8 + radius * 2;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.invalidateProjectedPointBuckets();
  }

  setSelection(result) {
    this.currentSelection = result;
    this.selectionPointLookup = this.buildSelectionPointLookup(result);
    this.renderSelectionLayers();
  }

  updateHoverFromScreen(screenX, screenY, options = {}) {
    const hint = this.findPointHint(screenX, screenY, options);
    if (!hint) {
      this.clearHover();
      return null;
    }

    const point = this.getPointById(hint.pointId);
    if (!point) {
      this.clearHover();
      return null;
    }

    this.setHoverPoint(point);
    return {
      pointId: point.point_id,
      xyz: point.xyz,
      pixelDistance: hint.pixelDistance,
    };
  }

  getPointById(pointId) {
    return this.pointLookup.get(pointId) ?? null;
  }

  getSelectionPointById(pointId) {
    return this.selectionPointLookup.get(pointId) ?? this.getPointById(pointId);
  }

  getInteractionElement() {
    return this.renderer.domElement;
  }

  setInteractionEnabled(enabled) {
    this.renderer.domElement.style.pointerEvents = enabled ? "auto" : "none";
    this.renderer.domElement.style.opacity = enabled ? "1" : "0.7";
    this.controls.enabled = enabled;
  }

  setHoverPointById(pointId) {
    const point = this.getPointById(pointId);
    if (!point) {
      this.clearHover();
      return null;
    }
    this.setHoverPoint(point);
    return point;
  }

  focusOnPointById(pointId) {
    const point = this.getSelectionPointById(pointId);
    if (!point) {
      return false;
    }

    const nextTarget = new THREE.Vector3(...point.xyz);
    const cameraOffset = this.camera.position.clone().sub(this.controls.target);
    this.controls.target.copy(nextTarget);
    this.camera.position.copy(nextTarget.clone().add(cameraOffset));
    this.controls.update();
    this.invalidateProjectedPointBuckets();
    this.invalidatePickBuffer();
    return true;
  }

  setClassificationFilter(filterValue) {
    this.classificationFilter = filterValue || "all";
    this.renderSelectionLayers();
  }

  renderSelectionLayers() {
    if (!this.currentSelection || !shouldDisplaySelection(this.currentSelection, this.classificationFilter)) {
      this.resetSelectionLayers();
      return;
    }

    const result = this.currentSelection;
    const palette = paletteForLabel(result.classification?.label);
    const roiPositions = this.collectSelectionPositions(result.roi?.points, result.roi?.point_ids);
    const maskPositions = this.collectSelectionPositions(result.mask?.points, result.mask?.point_ids);

    this.highlightGeometry.dispose();
    this.highlightGeometry = new THREE.BufferGeometry();
    this.highlightGeometry.setAttribute("position", new THREE.Float32BufferAttribute(roiPositions, 3));
    this.highlightPoints.geometry = this.highlightGeometry;
    this.highlightMaterial.color.set(palette.roiColor);

    this.maskGeometry.dispose();
    this.maskGeometry = new THREE.BufferGeometry();
    this.maskGeometry.setAttribute("position", new THREE.Float32BufferAttribute(maskPositions, 3));
    this.maskPoints.geometry = this.maskGeometry;
    this.maskMaterial.color.set(palette.maskColor);

    const selectionSeedPointId = result.mask?.seed_point_id ?? result.pick.point_id;
    const selectionSeedPoint = this.getSelectionPointById(selectionSeedPointId);
    const selectedPosition = selectionSeedPoint?.xyz ?? result.pick.xyz;
    this.selectedGeometry.dispose();
    this.selectedGeometry = new THREE.BufferGeometry();
    this.selectedGeometry.setAttribute("position", new THREE.Float32BufferAttribute(selectedPosition, 3));
    this.selectedPoint.geometry = this.selectedGeometry;
    this.selectedMaterial.color.set(palette.pickColor);

    const positivePositions = this.collectSelectionPositions(
      result.refinement?.positive_points,
      result.refinement?.positive_point_ids ?? result.mask?.positive_point_ids,
    );

    this.positiveGeometry.dispose();
    this.positiveGeometry = new THREE.BufferGeometry();
    this.positiveGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positivePositions, 3));
    this.positivePoints.geometry = this.positiveGeometry;

    const negativePositions = this.collectSelectionPositions(
      result.refinement?.negative_points,
      result.refinement?.negative_point_ids ?? result.mask?.negative_point_ids,
    );

    this.negativeGeometry.dispose();
    this.negativeGeometry = new THREE.BufferGeometry();
    this.negativeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(negativePositions, 3));
    this.negativePoints.geometry = this.negativeGeometry;

    this.updateCandidateBox(result.candidate_box, palette.boxColor);
  }

  clearSelection() {
    this.currentSelection = null;
    this.selectionPointLookup = new Map();
    this.resetSelectionLayers();
  }

  setHoverPoint(point) {
    if (this.currentHoverPointId === point.point_id) {
      this.container.classList.add("is-point-hovered");
      return;
    }

    this.currentHoverPointId = point.point_id;
    this.hoverGeometry.dispose();
    this.hoverGeometry = new THREE.BufferGeometry();
    this.hoverGeometry.setAttribute("position", new THREE.Float32BufferAttribute(point.xyz, 3));
    this.hoverPoint.geometry = this.hoverGeometry;
    this.container.classList.add("is-point-hovered");
  }

  clearHover() {
    this.currentHoverPointId = null;
    this.hoverGeometry.dispose();
    this.hoverGeometry = new THREE.BufferGeometry();
    this.hoverPoint.geometry = this.hoverGeometry;
    this.container.classList.remove("is-point-hovered");
  }

  resetSelectionLayers() {
    this.highlightGeometry.dispose();
    this.highlightGeometry = new THREE.BufferGeometry();
    this.highlightPoints.geometry = this.highlightGeometry;

    this.maskGeometry.dispose();
    this.maskGeometry = new THREE.BufferGeometry();
    this.maskPoints.geometry = this.maskGeometry;

    this.selectedGeometry.dispose();
    this.selectedGeometry = new THREE.BufferGeometry();
    this.selectedPoint.geometry = this.selectedGeometry;
    this.positiveGeometry.dispose();
    this.positiveGeometry = new THREE.BufferGeometry();
    this.positivePoints.geometry = this.positiveGeometry;
    this.negativeGeometry.dispose();
    this.negativeGeometry = new THREE.BufferGeometry();
    this.negativePoints.geometry = this.negativeGeometry;
    this.candidateGeometry.dispose();
    this.candidateGeometry = new THREE.BufferGeometry();
    this.candidateOutline.geometry = this.candidateGeometry;
    this.candidateOutline.visible = false;
  }

  buildSelectionPointLookup(result) {
    const lookup = new Map();
    for (const point of [
      ...(result?.roi?.points ?? []),
      ...(result?.mask?.points ?? []),
      ...(result?.refinement?.positive_points ?? []),
      ...(result?.refinement?.negative_points ?? []),
    ]) {
      if (!point || point.point_id == null || !Array.isArray(point.xyz)) {
        continue;
      }
      lookup.set(point.point_id, point);
    }
    return lookup;
  }

  collectSelectionPositions(explicitPoints = [], fallbackPointIds = []) {
    if (Array.isArray(explicitPoints) && explicitPoints.length > 0) {
      const positions = [];
      for (const point of explicitPoints) {
        if (Array.isArray(point?.xyz)) {
          positions.push(...point.xyz);
        }
      }
      return positions;
    }

    const positions = [];
    for (const pointId of fallbackPointIds ?? []) {
      const point = this.getSelectionPointById(pointId);
      if (point) {
        positions.push(...point.xyz);
      }
    }
    return positions;
  }

  getCameraPayload() {
    const { clientWidth: width, clientHeight: height } = this.container;
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const fy = height / (2 * Math.tan(verticalFov / 2));
    const fx = fy * this.camera.aspect;

    return {
      origin: this.camera.position.toArray(),
      target: this.controls.target.toArray(),
      up: this.camera.up.toArray(),
      width,
      height,
      fx,
      fy,
      cx: width / 2,
      cy: height / 2,
    };
  }

  handleCameraChange() {
    this.invalidateProjectedPointBuckets();
    this.invalidatePickBuffer();
  }

  invalidateProjectedPointBuckets() {
    this.projectedPointBuckets = new Map();
    this.projectedPointBucketsSignature = "";
    this.projectedPointMinCameraDepth = Number.POSITIVE_INFINITY;
  }

  invalidatePickBuffer() {
    this.pickBufferNeedsRender = true;
  }

  syncPickRenderTarget() {
    this.renderer.getDrawingBufferSize(this.drawingBufferSize);
    const width = Math.max(1, Math.floor(this.drawingBufferSize.x));
    const height = Math.max(1, Math.floor(this.drawingBufferSize.y));

    if (!this.pickRenderTarget || this.pickRenderTarget.width !== width || this.pickRenderTarget.height !== height) {
      if (this.pickRenderTarget) {
        this.pickRenderTarget.dispose();
      }
      this.pickRenderTarget = new THREE.WebGLRenderTarget(width, height, {
        depthBuffer: true,
        stencilBuffer: false,
        magFilter: THREE.NearestFilter,
        minFilter: THREE.NearestFilter,
        type: THREE.UnsignedByteType,
      });
      this.pickReadBuffer = new Uint8Array(0);
      this.invalidatePickBuffer();
    }

    this.pickMaterial.uniforms.size.value = this.baseMaterial.size;
    this.pickMaterial.uniforms.scale.value = height / 2;
  }

  renderPickBufferIfNeeded() {
    if (!this.pickRenderTarget || !this.pickBufferNeedsRender) {
      return;
    }

    const previousTarget = this.renderer.getRenderTarget();
    const previousClearAlpha = this.renderer.getClearAlpha();
    this.renderer.getClearColor(this.clearColorValue);

    this.renderer.setRenderTarget(this.pickRenderTarget);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.pickScene, this.camera);
    this.renderer.setRenderTarget(previousTarget);
    this.renderer.setClearColor(this.clearColorValue, previousClearAlpha);

    this.pickBufferNeedsRender = false;
  }

  readPointHintFromPickBuffer(screenX, screenY, maxPixelDistance = 28, exactPixelMatchOnly = false) {
    if (this.points.length === 0) {
      return null;
    }

    this.syncPickRenderTarget();
    if (!this.pickRenderTarget) {
      return null;
    }
    this.renderPickBufferIfNeeded();

    const clientWidth = Math.max(this.container.clientWidth, 1);
    const clientHeight = Math.max(this.container.clientHeight, 1);
    const bufferWidth = this.pickRenderTarget.width;
    const bufferHeight = this.pickRenderTarget.height;
    const scaleX = bufferWidth / clientWidth;
    const scaleY = bufferHeight / clientHeight;
    const targetX = clampNumber(Math.round(screenX * scaleX), 0, bufferWidth - 1);
    const targetYFromTop = clampNumber(Math.round(screenY * scaleY), 0, bufferHeight - 1);
    const targetY = bufferHeight - 1 - targetYFromTop;
    const radiusX = exactPixelMatchOnly ? 0 : Math.max(0, Math.ceil(maxPixelDistance * scaleX));
    const radiusY = exactPixelMatchOnly ? 0 : Math.max(0, Math.ceil(maxPixelDistance * scaleY));
    const originX = Math.max(0, targetX - radiusX);
    const originY = Math.max(0, targetY - radiusY);
    const endX = Math.min(bufferWidth - 1, targetX + radiusX);
    const endY = Math.min(bufferHeight - 1, targetY + radiusY);
    const readWidth = endX - originX + 1;
    const readHeight = endY - originY + 1;
    const requiredBufferLength = readWidth * readHeight * 4;

    if (this.pickReadBuffer.length !== requiredBufferLength) {
      this.pickReadBuffer = new Uint8Array(requiredBufferLength);
    }

    this.renderer.readRenderTargetPixels(
      this.pickRenderTarget,
      originX,
      originY,
      readWidth,
      readHeight,
      this.pickReadBuffer,
    );

    const hit = findNearestPickHit({
      pixels: this.pickReadBuffer,
      width: readWidth,
      height: readHeight,
      originX,
      originY,
      targetX,
      targetY,
    });
    if (!hit) {
      return null;
    }

    const pointId = this.pickIndexToPointId[hit.pickIndex - 1];
    if (pointId == null) {
      return null;
    }

    return {
      pointId,
      pixelDistance: hit.pixelDistance / Math.max(scaleX, scaleY, 1),
      source: "pick-buffer",
    };
  }

  rebuildProjectedPointBuckets(signature = this.computeProjectedPointBucketsSignature()) {
    this.projectedPointBuckets = new Map();
    this.projectedPointMinCameraDepth = Number.POSITIVE_INFINITY;
    this.projectedPointBucketsSignature = signature || "";
    if (!signature) {
      return;
    }

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    for (const point of this.points) {
      this.projectedVector.set(...point.xyz).project(this.camera);
      if (this.projectedVector.z < -1 || this.projectedVector.z > 1) {
        continue;
      }

      const projectedX = (this.projectedVector.x * 0.5 + 0.5) * width;
      const projectedY = (-this.projectedVector.y * 0.5 + 0.5) * height;
      this.cameraSpaceVector.set(...point.xyz).applyMatrix4(this.camera.matrixWorldInverse);
      const cameraDepth = Math.max(1e-6, -this.cameraSpaceVector.z);
      this.projectedPointMinCameraDepth = Math.min(this.projectedPointMinCameraDepth, cameraDepth);

      const bucketX = Math.floor(projectedX / this.projectedPointBucketSizePx);
      const bucketY = Math.floor(projectedY / this.projectedPointBucketSizePx);
      const bucketKey = projectedPointBucketKey(bucketX, bucketY);
      const bucketEntries = this.projectedPointBuckets.get(bucketKey) ?? [];
      bucketEntries.push({
        pointId: point.point_id,
        screenX: projectedX,
        screenY: projectedY,
        depth: this.projectedVector.z,
        cameraDepth,
      });
      this.projectedPointBuckets.set(bucketKey, bucketEntries);
    }
  }

  computeProjectedPointBucketsSignature() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width <= 0 || height <= 0 || this.points.length === 0) {
      return "";
    }

    return [
      width,
      height,
      this.points.length,
      ...this.camera.position.toArray(),
      ...this.camera.quaternion.toArray(),
      ...this.controls.target.toArray(),
    ]
      .map((value, index) => (index < 3 ? String(value) : Number(value).toFixed(5)))
      .join("|");
  }

  ensureProjectedPointBuckets() {
    const signature = this.computeProjectedPointBucketsSignature();
    if (!signature) {
      this.invalidateProjectedPointBuckets();
      return;
    }
    if (signature !== this.projectedPointBucketsSignature) {
      this.rebuildProjectedPointBuckets(signature);
    }
  }

  collectProjectedBucketCandidates(screenX, screenY, maxPixelDistance = 28) {
    this.ensureProjectedPointBuckets();
    if (this.projectedPointBuckets.size === 0) {
      return [];
    }

    const maxHitRadiusPx = Number.isFinite(this.projectedPointMinCameraDepth)
      ? this.estimateProjectedHitRadiusPx(this.projectedPointMinCameraDepth)
      : 0;
    const bucketRadius = Math.max(
      1,
      Math.ceil((maxPixelDistance + maxHitRadiusPx + 2) / this.projectedPointBucketSizePx),
    );
    const centerBucketX = Math.floor(screenX / this.projectedPointBucketSizePx);
    const centerBucketY = Math.floor(screenY / this.projectedPointBucketSizePx);
    const candidates = [];

    for (let bucketX = centerBucketX - bucketRadius; bucketX <= centerBucketX + bucketRadius; bucketX += 1) {
      for (let bucketY = centerBucketY - bucketRadius; bucketY <= centerBucketY + bucketRadius; bucketY += 1) {
        const bucketEntries = this.projectedPointBuckets.get(projectedPointBucketKey(bucketX, bucketY));
        if (bucketEntries) {
          candidates.push(...bucketEntries);
        }
      }
    }

    return candidates;
  }

  findPointHint(screenX, screenY, maxPixelDistance = 28) {
    const resolvedOptions = typeof maxPixelDistance === "number"
      ? { maxPixelDistance }
      : (maxPixelDistance ?? {});
    const maxPixelDistanceValue = Number.isFinite(resolvedOptions.maxPixelDistance)
      ? resolvedOptions.maxPixelDistance
      : 28;
    const maxDistanceToRay = Number.isFinite(resolvedOptions.maxDistanceToRay)
      ? Math.max(0, resolvedOptions.maxDistanceToRay)
      : null;
    const exactPixelMatchOnly = resolvedOptions.exactPixelMatchOnly === true;
    const pickBufferHint = this.readPointHintFromPickBuffer(
      screenX,
      screenY,
      maxPixelDistanceValue,
      exactPixelMatchOnly,
    );
    if (pickBufferHint) {
      return pickBufferHint;
    }
    if (exactPixelMatchOnly) {
      return null;
    }
    let bestPixelMatch = null;
    let bestRayMatch = null;
    const rayDirection = maxDistanceToRay !== null
      ? this.getScreenRayDirection(screenX, screenY)
      : null;
    const candidates = this.collectProjectedBucketCandidates(screenX, screenY, maxPixelDistanceValue);

    for (const candidate of candidates) {
      const point = this.getPointById(candidate.pointId);
      if (!point) {
        continue;
      }

      const hitHalfSizePx = this.estimateProjectedHitRadiusPx(candidate.cameraDepth);
      const hitMetrics = computeSquareHitMetrics({
        screenX,
        screenY,
        pointScreenX: candidate.screenX,
        pointScreenY: candidate.screenY,
        halfSizePx: hitHalfSizePx,
      });

      if (hitMetrics.effectiveSquareDistance > maxPixelDistanceValue) {
        continue;
      }

      const pixelCandidateKey = buildHintCandidateKey({
        effectiveSquareDistance: hitMetrics.effectiveSquareDistance,
        pixelDistance: hitMetrics.pixelDistance,
        depth: candidate.depth,
        pointId: candidate.pointId,
        insideSquare: hitMetrics.insideSquare,
      });
      const bestPixelKey = bestPixelMatch
        ? bestPixelMatch.key
        : null;
      if (!bestPixelMatch || compareHintKeys(pixelCandidateKey, bestPixelKey) < 0) {
        bestPixelMatch = {
          pointId: candidate.pointId,
          key: pixelCandidateKey,
          depth: candidate.depth,
          effectivePixelDistance: hitMetrics.effectiveSquareDistance,
          pixelDistance: hitMetrics.pixelDistance,
        };
      }

      if (rayDirection) {
        this.relativeVector.set(...point.xyz).sub(this.camera.position);
        const projection = this.relativeVector.dot(rayDirection);
        if (projection < 0) {
          continue;
        }
        const perpendicularSq = Math.max(0, this.relativeVector.lengthSq() - projection * projection);
        const distanceToRay = Math.sqrt(perpendicularSq);
        if (distanceToRay > maxDistanceToRay) {
          continue;
        }

        const rayCandidateKey = buildHintCandidateKey({
          effectiveSquareDistance: hitMetrics.effectiveSquareDistance,
          pixelDistance: hitMetrics.pixelDistance,
          depth: candidate.depth,
          projectionLength: projection,
          distanceToRay,
          pointId: candidate.pointId,
          insideSquare: hitMetrics.insideSquare,
        });
        const bestRayKey = bestRayMatch
          ? bestRayMatch.key
          : null;
        if (!bestRayMatch || compareHintKeys(rayCandidateKey, bestRayKey) < 0) {
          bestRayMatch = {
            pointId: candidate.pointId,
            key: rayCandidateKey,
            effectivePixelDistance: hitMetrics.effectiveSquareDistance,
            projectionLength: projection,
            distanceToRay,
            pixelDistance: hitMetrics.pixelDistance,
          };
        }
      }
    }

    return bestRayMatch ?? bestPixelMatch;
  }

  estimatePointHitRadiusPx(point) {
    this.cameraSpaceVector.set(...point.xyz).applyMatrix4(this.camera.matrixWorldInverse);
    return this.estimateProjectedHitRadiusPx(-this.cameraSpaceVector.z);
  }

  estimateProjectedHitRadiusPx(cameraDepth) {
    const depth = Math.max(1e-6, cameraDepth);
    const projectionScale = (this.renderer.domElement.height * this.camera.projectionMatrix.elements[5]) / 2;
    const pointSizeDevicePx = this.baseMaterial.size * projectionScale / depth;
    const pointSizeCssPx = pointSizeDevicePx / Math.max(this.renderer.getPixelRatio(), 1);
    return Math.max(1.5, pointSizeCssPx / 2);
  }

  getScreenRayDirection(screenX, screenY) {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    const ndcX = (screenX / width) * 2 - 1;
    const ndcY = -(screenY / height) * 2 + 1;
    this.raySampleVector.set(ndcX, ndcY, 0.5).unproject(this.camera);
    return this.rayDirectionVector.copy(this.raySampleVector).sub(this.camera.position).normalize();
  }

  handleResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.invalidateProjectedPointBuckets();
    this.invalidatePickBuffer();
  }

  animate = () => {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.animationFrame = window.requestAnimationFrame(this.animate);
  };

  destroy() {
    window.removeEventListener("resize", this.handleResize);
    this.controls.removeEventListener("change", this.handleCameraChange);
    window.cancelAnimationFrame(this.animationFrame);
    this.controls.dispose();
    this.pickGeometry.dispose();
    this.pickMaterial.dispose();
    if (this.pickRenderTarget) {
      this.pickRenderTarget.dispose();
      this.pickRenderTarget = null;
    }
    this.roundPointTexture.dispose();
    this.renderer.dispose();
  }

  applyPointSizes() {
    const scaledBase = this.basePointSize * this.pointSizeScale;
    this.baseMaterial.size = scaledBase;
    this.highlightMaterial.size = Math.max(scaledBase * 1.75, 0.07 * this.pointSizeScale);
    this.maskMaterial.size = Math.max(scaledBase * 2.1, 0.09 * this.pointSizeScale);
    this.selectedMaterial.size = Math.max(scaledBase * 2.7, 0.12 * this.pointSizeScale);
    this.hoverMaterial.size = Math.max(scaledBase * 3.2, 0.14 * this.pointSizeScale);
    this.positiveMaterial.size = Math.max(scaledBase * 2.5, 0.11 * this.pointSizeScale);
    this.negativeMaterial.size = Math.max(scaledBase * 2.4, 0.11 * this.pointSizeScale);
    this.pickMaterial.uniforms.size.value = this.baseMaterial.size;
    this.invalidatePickBuffer();
  }

  updateCandidateBox(candidateBox, color) {
    if (!candidateBox) {
      this.candidateOutline.visible = false;
      return;
    }

    const lineVertices = buildCandidateLineVertices(candidateBox);
    this.candidateGeometry.dispose();
    this.candidateGeometry = new THREE.BufferGeometry();
    this.candidateGeometry.setAttribute("position", new THREE.Float32BufferAttribute(lineVertices, 3));
    this.candidateOutline.geometry = this.candidateGeometry;
    this.candidateMaterial.color.set(color);
    this.candidateOutline.visible = lineVertices.length > 0;
  }
}

function recommendedPointSize(scenePayload) {
  const [minX, minY, minZ] = scenePayload.bounds.min;
  const [maxX, maxY, maxZ] = scenePayload.bounds.max;
  const diagonal = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);

  if (scenePayload.point_count <= 200) {
    return Math.max(diagonal * 0.018, 0.08);
  }
  if (scenePayload.point_count <= 2000) {
    return Math.max(diagonal * 0.01, 0.045);
  }
  if (scenePayload.point_count <= 50000) {
    return Math.max(diagonal * 0.0055, 0.018);
  }
  if (scenePayload.point_count <= 250000) {
    return Math.max(diagonal * 0.0045, 0.014);
  }
  return Math.max(diagonal * 0.0038, 0.012);
}

function recommendedPixelRatio(pointCount) {
  const devicePixelRatio = window.devicePixelRatio || 1;
  if (pointCount >= 800000) {
    return Math.min(devicePixelRatio, 1.0);
  }
  if (pointCount >= 300000) {
    return Math.min(devicePixelRatio, 1.1);
  }
  if (pointCount >= 120000) {
    return Math.min(devicePixelRatio, 1.3);
  }
  return Math.min(devicePixelRatio, 1.65);
}

function createRoundPointTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Unable to create round point texture");
  }

  const center = size / 2;
  const gradient = context.createRadialGradient(center, center, 3, center, center, center);
  gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
  gradient.addColorStop(0.7, "rgba(255, 255, 255, 1)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  context.clearRect(0, 0, size, size);
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function createRoundPointsMaterial(texture, overrides = {}) {
  return new THREE.PointsMaterial({
    sizeAttenuation: true,
    transparent: true,
    depthWrite: false,
    alphaTest: 0.22,
    alphaMap: texture,
    ...overrides,
  });
}

function createPickMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      size: { value: 0.05 },
      scale: { value: 1 },
    },
    vertexColors: true,
    depthTest: true,
    depthWrite: true,
    transparent: false,
    blending: THREE.NoBlending,
    vertexShader: `
      uniform float size;
      uniform float scale;
      varying vec3 vColor;

      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (scale / max(1e-6, -mvPosition.z));
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;

      void main() {
        vec2 centered = gl_PointCoord - vec2(0.5);
        if (dot(centered, centered) > 0.25) {
          discard;
        }
        gl_FragColor = vec4(vColor, 1.0);
      }
    `,
  });
}

function paletteForLabel(label) {
  if (label === "door") {
    return {
      roiColor: "#ffcc8f",
      maskColor: "#ff8b5b",
      pickColor: "#ffd57f",
      boxColor: "#ff8b5b",
    };
  }
  if (label === "window") {
    return {
      roiColor: "#b4ddff",
      maskColor: "#63d0ff",
      pickColor: "#8cc4ff",
      boxColor: "#63d0ff",
    };
  }
  return {
    roiColor: "#ffb35c",
    maskColor: "#86f0ce",
    pickColor: "#8cc4ff",
    boxColor: "#86f0ce",
  };
}

function shouldDisplaySelection(result, filterValue) {
  if (!result) {
    return false;
  }
  if (filterValue === "all") {
    return true;
  }
  return (result.classification?.label ?? "unknown") === filterValue;
}

function buildCandidateLineVertices(candidateBox) {
  const corners = Array.isArray(candidateBox?.corners) && candidateBox.corners.length === 8
    ? candidateBox.corners
    : cornersFromBounds(candidateBox?.min, candidateBox?.max);

  if (corners.length !== 8) {
    return [];
  }

  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const vertices = [];

  for (const [startIndex, endIndex] of edges) {
    vertices.push(...corners[startIndex], ...corners[endIndex]);
  }

  return vertices;
}

function compareHintKeys(left, right) {
  if (!right) {
    return -1;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] < right[index]) {
      return -1;
    }
    if (left[index] > right[index]) {
      return 1;
    }
  }
  return 0;
}

function projectedPointBucketKey(bucketX, bucketY) {
  return `${bucketX}:${bucketY}`;
}

function clampNumber(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function cornersFromBounds(min, max) {
  if (!Array.isArray(min) || !Array.isArray(max) || min.length !== 3 || max.length !== 3) {
    return [];
  }

  const [minX, minY, minZ] = min;
  const [maxX, maxY, maxZ] = max;
  return [
    [minX, minY, minZ],
    [maxX, minY, minZ],
    [maxX, maxY, minZ],
    [minX, maxY, minZ],
    [minX, minY, maxZ],
    [maxX, minY, maxZ],
    [maxX, maxY, maxZ],
    [minX, maxY, maxZ],
  ];
}

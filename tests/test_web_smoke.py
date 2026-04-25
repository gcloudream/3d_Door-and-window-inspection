import unittest
from pathlib import Path


class FrontendSmokeTests(unittest.TestCase):
    def test_index_references_app_entry(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn("app-shell", html)
        self.assertIn("canvas-shell", html)
        self.assertIn("/web/app.js", html)

    def test_index_declares_import_map_for_three(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('type="importmap"', html)
        self.assertIn('"three"', html)
        self.assertIn('"three/addons/"', html)

    def test_index_exposes_scene_upload_control(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('id="scene-file"', html)
        self.assertIn('id="scene-density"', html)
        self.assertIn('<option value="300000" selected>', html)
        self.assertIn("选择点云文件", html)
        self.assertIn("点云密度", html)
        self.assertIn(".pcd", html)

    def test_index_exposes_mask_result_fields(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('id="mask-point-count"', html)
        self.assertIn('id="mask-confidence"', html)
        self.assertIn('id="candidate-label"', html)
        self.assertIn('id="candidate-confidence"', html)
        self.assertIn('id="candidate-filter"', html)
        self.assertIn("仅看门候选", html)
        self.assertIn("仅看窗候选", html)

    def test_index_exposes_candidate_geometry_fields(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('id="candidate-box-shape"', html)
        self.assertIn('id="candidate-box-anchor"', html)
        self.assertIn("候选框类型", html)
        self.assertIn("锚定方式", html)

    def test_index_exposes_negative_refine_controls(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('id="refine-mode-toggle"', html)
        self.assertIn('id="augment-mode-toggle"', html)
        self.assertIn('id="negative-point-count"', html)
        self.assertIn('id="positive-point-count"', html)
        self.assertIn("排除模式", html)
        self.assertIn("负样本", html)
        self.assertIn("补充模式", html)
        self.assertIn("正样本", html)

    def test_index_exposes_point_size_control(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('id="point-size-scale"', html)
        self.assertIn('id="point-size-scale-value"', html)
        self.assertIn("点大小", html)

    def test_scene_view_supports_runtime_point_size_updates(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/scene-view.js").read_text()

        self.assertIn("setPointSizeScale", js)
        self.assertIn("applyPointSizes", js)
        self.assertIn("getInteractionElement()", js)
        self.assertIn("createRoundPointTexture", js)
        self.assertIn("alphaMap", js)

    def test_index_exposes_hover_feedback_fields(self) -> None:
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn('id="hover-indicator"', html)
        self.assertIn('id="hover-point-id"', html)
        self.assertIn("悬停点", html)

    def test_scene_view_supports_hover_feedback(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/scene-view.js").read_text()

        self.assertIn("updateHoverFromScreen", js)
        self.assertIn("setHoverPoint", js)
        self.assertIn("clearHover", js)
        self.assertIn("selectionPointLookup", js)
        self.assertIn("buildSelectionPointLookup", js)

    def test_app_updates_hover_feedback_locally_without_preview_round_trip(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/app.js").read_text()

        self.assertIn("sceneView.updateHoverFromScreen", js)
        self.assertIn("/api/scene/reload", js)
        self.assertIn("flushHoverFeedback", js)
        self.assertIn("locked_point_id", js)
        self.assertIn("result.mask.points", js)
        self.assertNotIn("/api/pick-preview", js)
        self.assertNotIn("requestHoverPreview", js)

    def test_scene_view_accounts_for_rendered_point_radius_in_hover_pick(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/scene-view.js").read_text()

        self.assertIn("estimatePointHitRadiusPx", js)
        self.assertIn("effectivePixelDistance", js)

    def test_scene_view_builds_projected_bucket_index_for_hover_pick(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/scene-view.js").read_text()

        self.assertIn("projectedPointBuckets", js)
        self.assertIn("rebuildProjectedPointBuckets", js)
        self.assertIn("invalidateProjectedPointBuckets", js)
        self.assertIn("collectProjectedBucketCandidates", js)

    def test_scene_view_supports_pick_buffer_pixel_readback(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/scene-view.js").read_text()

        self.assertIn("pickRenderTarget", js)
        self.assertIn("readPointHintFromPickBuffer", js)
        self.assertIn("syncPickRenderTarget", js)
        self.assertIn("gl_PointCoord", js)
        self.assertIn("discard;", js)

    def test_controls_sidebar_fills_right_column_after_telemetry_removal(self) -> None:
        css = Path("/Users/gengchen/Desktop/3d/web/styles.css").read_text()

        self.assertIn(".controls-card {", css)
        self.assertIn("height: 100%;", css)
        self.assertIn("overflow-y: auto;", css)

    def test_layout_prioritizes_large_viewer_with_compact_controls(self) -> None:
        css = Path("/Users/gengchen/Desktop/3d/web/styles.css").read_text()

        self.assertIn("grid-template-columns: minmax(0, 1fr) minmax(264px, 304px);", css)
        self.assertIn("gap: 8px;", css)
        self.assertIn("padding: 9px;", css)

    def test_scene_view_optimizes_camera_drag_controls(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/scene-view.js").read_text()

        self.assertIn("this.controls.screenSpacePanning = true;", js)
        self.assertIn("this.controls.zoomToCursor = true;", js)
        self.assertIn("this.controls.enableDamping = true;", js)
        self.assertIn("this.controls.dampingFactor = 0.08;", js)
        self.assertIn("this.controls.rotateSpeed = 0.46;", js)
        self.assertIn("this.controls.panSpeed = 0.92;", js)
        self.assertIn("this.controls.minPolarAngle = 0.08;", js)
        self.assertIn("this.controls.maxPolarAngle = Math.PI - 0.08;", js)
        self.assertIn("focusOnPointById", js)

    def test_app_recenters_orbit_after_successful_selection(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/app.js").read_text()
        html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()

        self.assertIn("sceneView.focusOnPointById(getSelectionSeedPointId(result));", js)
        self.assertIn("右键平移", html)
        self.assertIn("选中后围绕目标旋转", html)

    def test_app_suspends_hover_and_click_selection_while_dragging_view(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/app.js").read_text()

        self.assertIn("const DRAG_CLICK_SUPPRESSION_PX = 5;", js)
        self.assertIn("const interactionElement = sceneView.getInteractionElement();", js)
        self.assertIn("isViewerDragging", js)
        self.assertIn("suppressNextClick", js)
        self.assertIn("setPointerCapture", js)
        self.assertIn("releasePointerCapture", js)
        self.assertIn("clearPendingHoverFeedback();", js)
        self.assertIn("if (suppressNextClick)", js)
        self.assertIn('interactionElement.addEventListener("pointerdown"', js)
        self.assertIn('interactionElement.addEventListener("pointermove"', js)
        self.assertIn('interactionElement.addEventListener("click"', js)
        self.assertNotIn('viewerElement.setPointerCapture', js)

    def test_viewer_disables_context_menu_for_right_drag_pan(self) -> None:
        js = Path("/Users/gengchen/Desktop/3d/web/app.js").read_text()
        css = Path("/Users/gengchen/Desktop/3d/web/styles.css").read_text()

        self.assertIn('interactionElement.addEventListener("contextmenu"', js)
        self.assertIn("event.preventDefault();", js)
        self.assertIn("cursor: grab;", css)
        self.assertIn("cursor: grabbing;", css)


if __name__ == "__main__":
    unittest.main()

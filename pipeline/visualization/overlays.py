from cams import show_cam_overlay, show_full_ct_cam_overlay


class VolumeOverlayViewer:
    def show_stage1_patch_cam(self, patch, explanation, title, cam_threshold=0.35):
        show_cam_overlay(
            patch,
            explanation.cam,
            mask=explanation.mask,
            title=title,
            cam_threshold=cam_threshold,
        )

    def show_stage1_patient_cams(self, context, cam_points, title):
        show_full_ct_cam_overlay(
            context.ct,
            context.seg,
            cam_points,
            title=title,
            cam_threshold=None,
            show_seg_context=False,
        )

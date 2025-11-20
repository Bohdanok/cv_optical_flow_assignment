from pathlib import Path
from typing import Any, Tuple, Optional, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from dataclasses import dataclass

CALIBRATION_FRAME_COUNT = 500

@dataclass
class TrajectoryResult:
    timestamps: List[float]
    positions: np.ndarray
    inliers: List[int]
    frame_indices: List[int]

class TrajectoryVideoWriter:
    """Encapsulates video writing with trajectory overlay."""
    
    def __init__(
        self,
        output_path: Optional[Path],
        fps: float,
        frame_size: Tuple[int, int],
        draw_thickness: int = 1
    ):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.draw_thickness = draw_thickness
        self.writer = None
        self.absolute_index = 0
        
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") if output_path.suffix.lower() in {".mp4", ".m4v"} else cv2.VideoWriter_fourcc(*"avc1")
            self.writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), frame_size, True)
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open writer for {output_path}")
    
    def write_frame_with_overlay(
        self,
        frame: np.ndarray,
        pts1: Optional[np.ndarray],
        pts2: Optional[np.ndarray],
        inlier_mask: Optional[np.ndarray],
        status_msg: str,
        status_color: Tuple[int, int, int]
    ) -> None:
        """Draw tracks and write frame with status message."""
        if self.writer is None:
            return
        
        frame_draw = frame.copy()
        if pts1 is not None and pts2 is not None:
            draw_tracks(frame_draw, pts1, pts2, inlier_mask, self.draw_thickness)
        
        cv2.putText(frame_draw, f"Frame {self.absolute_index} | {status_msg}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        self.writer.write(frame_draw)
    
    def increment_frame(self) -> None:
        """Increment the absolute frame index."""
        self.absolute_index += 1
    
    def release(self) -> None:
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()

def plot_trajectory(result: Any) -> None:
    if getattr(result, "positions", None) is None or len(result.positions) == 0:
        print("No trajectory to plot")
        return

    positions = result.positions

    _, axes = plt.subplots(1, 2, figsize=(22, 6))

    axes[0].plot(positions[:, 0], positions[:, 1], "-o", markersize=2)
    axes[0].set_title("XY top-down (relative)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(result.frame_indices, result.inliers, ".-")
    axes[1].set_title("Homography inliers per frame")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("# Inliers")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def write_video_with_trajectory(
    video_path: Path,
    traj_result: Any,
    output_path: Path,
    stride: int = 5,
    fig_traj_size: Tuple[float, float] = (6, 6),
    show_full_traj: bool = True,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    frame_indices = traj_result.frame_indices
    positions = traj_result.positions

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame from video.")
    h, w, _ = frame.shape

    traj_img_width = w
    traj_img_height = int(h * fig_traj_size[1] / fig_traj_size[0])

    output_w = max(w, traj_img_width)
    output_h = h + traj_img_height

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps / stride, 1.0), (output_w, output_h))

    plt.ioff()

    xs, ys = positions[:, 0], positions[:, 1]

    total_frames = len(frame_indices)
    frames_to_process = [vidx for vidx in range(total_frames) if vidx % stride == 0]

    for vidx in tqdm(frames_to_process, desc="Writing video with trajectory", unit="frame"):
        pose_idx = frame_indices[vidx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, pose_idx)
        ok, frame = cap.read()
        if not ok:
            break

        fig, ax = plt.subplots(figsize=fig_traj_size)
        if show_full_traj:
            ax.plot(xs, ys, "k-", alpha=0.25, label="Full traj")
            ax.plot(xs[: vidx + 1], ys[: vidx + 1], "b-")
            ax.scatter([xs[vidx]], [ys[vidx]], c="red", s=90, label="Current", zorder=5)
        else:
            ax.plot(xs[: vidx + 1], ys[: vidx + 1], "b-")
            ax.scatter([xs[vidx]], [ys[vidx]], c="red", s=90, label="Current", zorder=5)
        ax.set_title("Trajectory (XY, top-down)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        traj_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
        traj_img_rgb = cv2.cvtColor(traj_img, cv2.COLOR_RGBA2RGB)
        plt.close(fig)

        traj_img_resized = cv2.resize(traj_img_rgb, (w, traj_img_height), interpolation=cv2.INTER_AREA)
        traj_img_bgr = cv2.cvtColor(traj_img_resized, cv2.COLOR_RGB2BGR)

        if traj_img_bgr.shape[1] < output_w:
            pad_width = output_w - traj_img_bgr.shape[1]
            traj_img_bgr = cv2.copyMakeBorder(
                traj_img_bgr, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0
            )

        if frame.shape[1] < output_w:
            pad_width = output_w - frame.shape[1]
            frame_pad = cv2.copyMakeBorder(frame, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
        else:
            frame_pad = frame

        concat_img = np.concatenate((frame_pad, traj_img_bgr), axis=0)
        writer.write(concat_img)

    cap.release()
    writer.release()
    print(f"Output video saved to {output_path}")


def align_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape[0] < 2:
        raise ValueError("Need at least 2 points for alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = src_centered.T @ dst_centered / src_centered.shape[0]

    U, S, Vt = np.linalg.svd(covariance)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_src = np.sum(src_centered**2) / src_centered.shape[0]
    scale = np.sum(S) / var_src

    translation = dst_mean - scale * (R @ src_mean)

    return scale, R, translation


def load_and_prepare_data(
    gt_path: Path,
    vo_frame_indices: np.ndarray,
    vo_raw_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_df = pd.read_csv(gt_path).set_index("frame_number")

    frame_indices = np.array(vo_frame_indices, dtype=int)
    reindexed_gt = gt_df.reindex(frame_indices)[["x_meters", "y_meters"]].to_numpy()
    valid_mask = ~np.isnan(reindexed_gt).any(axis=1)

    filtered_frame_indices = frame_indices[valid_mask]
    filtered_vo_positions = vo_raw_positions[valid_mask]
    filtered_gt_positions = reindexed_gt[valid_mask]

    if len(filtered_vo_positions) < 2:
        raise ValueError("Not enough overlapping frames between VO and ground truth CSV.")

    return filtered_frame_indices, filtered_vo_positions, filtered_gt_positions


def draw_tracks(
    frame: np.ndarray,
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    thickness: int = 1
) -> None:
    """
    Draw tracks on frame.
    Green lines and endpoints for inliers.
    Red dots for non-inliers.
    """
    # prev_pts, curr_pts: shape (N,1,2) float32
    p_prev = prev_pts.reshape(-1, 2)
    p_curr = curr_pts.reshape(-1, 2)

    if inlier_mask is None:
        inlier_mask = np.ones((p_prev.shape[0],), dtype=bool)
    else:
        inlier_mask = inlier_mask.astype(bool).ravel()

    # Inliers
    pin_prev = p_prev[inlier_mask]
    pin_curr = p_curr[inlier_mask]
    for a, b in zip(pin_prev, pin_curr):
        p1 = (int(round(a[0])), int(round(a[1])))
        p2 = (int(round(b[0])), int(round(b[1])))
        cv2.line(frame, p1, p2, (0, 255, 0), thickness)
        cv2.circle(frame, p2, 2, (0, 255, 0), -1)

    # Outliers
    pout_curr = p_curr[~inlier_mask]
    for b in pout_curr:
        p2 = (int(round(b[0])), int(round(b[1])))
        cv2.circle(frame, p2, 2, (0, 0, 255), -1)


def plot_trajectories(gt_xy: np.ndarray, vo_xy: np.ndarray, title: str = "Trajectory Comparison") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground truth (CSV)", color="tab:orange", linewidth=2)
    ax.plot(vo_xy[:, 0], vo_xy[:, 1], label="VO (Aligned)", color="tab:green", linestyle="--", alpha=0.9)

    ax.plot(gt_xy[0, 0], gt_xy[0, 1], "o", color="red", markersize=8, label="Start (GT)")
    ax.plot(vo_xy[0, 0], vo_xy[0, 1], "x", color="red", markersize=10, mew=2, label="Start (VO)")

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.show()


def compute_path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))


def analyze_trajectory(
    traj_result: Any,
    gt_csv_path: Path,
    calibration_frame_count: int = CALIBRATION_FRAME_COUNT,
) -> None:
    try:
        vo_frames = np.asarray(traj_result.frame_indices)
        vo_positions_xy = np.asarray(traj_result.positions)[1:, :2]

        frame_indices, raw_positions_xy, gt_positions_xy = load_and_prepare_data(
            gt_csv_path, vo_frames, vo_positions_xy
        )

        calib_count = min(calibration_frame_count, len(raw_positions_xy))
        calib_src = raw_positions_xy[:calib_count]
        calib_dst = gt_positions_xy[:calib_count]

        scale, rotation, translation = align_similarity(calib_src, calib_dst)

        aligned_positions_xy = (scale * (raw_positions_xy @ rotation.T)) + translation

        calib_residuals = aligned_positions_xy[:calib_count] - calib_dst
        rms_error_calib = np.sqrt(np.mean(np.sum(calib_residuals**2, axis=1)))
        max_error_calib = np.max(np.linalg.norm(calib_residuals, axis=1))

        full_residuals = aligned_positions_xy - gt_positions_xy
        rms_error_full = np.sqrt(np.mean(np.sum(full_residuals**2, axis=1)))

        rotation_degrees = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))

        gt_path_length = compute_path_length(gt_positions_xy)
        aligned_vo_path_length = compute_path_length(aligned_positions_xy)

        print("--- Trajectory Alignment Report ---")
        print(f"Calibrated on first {calib_count} of {len(frame_indices)} overlapping frames.")
        print(
            f"\nSimilarity Alignment Parameters:\n"
            f"  Scale = {scale:.4f}\n"
            f"  Rotation = {rotation_degrees:.2f}°\n"
            f"  Translation = [{translation[0]:.2f}, {translation[1]:.2f}] m"
        )

        print(
            f"\nError (Calibration Segment):\n"
            f"  RMS = {rms_error_calib:.2f} m\n"
            f"  Max = {max_error_calib:.2f} m"
        )

        print(
            f"\nError (Full Trajectory):\n"
            f"  RMS = {rms_error_full:.2f} m\n"
        )
        print("---------------------------------")

        print(f"\nGround truth trajectory length: {gt_path_length:.2f} m")
        print(f"Aligned VO trajectory length:   {aligned_vo_path_length:.2f} m")

        plot_trajectories(
            gt_positions_xy,
            aligned_positions_xy,
            title=f"Trajectory Comparison (Aligned on first {calib_count} frames)",
        )

    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_csv_path}")
    except ValueError as exc:
        print(f"Error processing data: {exc}")
    except AttributeError:
        print(
            "Error: 'traj_result' object is missing expected attributes ('frame_indices' or 'positions')."
        )
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")


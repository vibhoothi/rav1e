// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::frame::Frame;
use crate::frame::Plane;
use crate::util::{CastFromPrimitive, Pixel};
use av_metrics::video::*;
//{ciede, psnr, psnr_hvs, ssim, FrameInfo, PlanarMetrics, PlaneData, ChromaSampling};

/// Calculates the PSNR for a `Frame` by comparing the original (uncompressed) to the compressed
/// version of the frame. Higher PSNR is better--PSNR is capped at 100 in order to avoid skewed
/// statistics from e.g. all black frames, which would otherwise show a PSNR of infinity.
///
/// See https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio for more details.
pub fn calculate_frame_psnr<T: Pixel>(
  original: &Frame<T>, compressed: &Frame<T>, bit_depth: usize,
) -> (f64, f64, f64) {
  (
    calculate_plane_psnr(
      &original.planes[0],
      &compressed.planes[0],
      bit_depth,
    ),
    calculate_plane_psnr(
      &original.planes[1],
      &compressed.planes[1],
      bit_depth,
    ),
    calculate_plane_psnr(
      &original.planes[2],
      &compressed.planes[2],
      bit_depth,
    ),
  )
}

/// Calculate the PSNR for a `Plane` by comparing the original (uncompressed) to the compressed
/// version.
fn calculate_plane_psnr<T: Pixel>(
  original: &Plane<T>, compressed: &Plane<T>, bit_depth: usize,
) -> f64 {
  let mse = calculate_plane_mse(original, compressed);
  if mse <= 0.000_000_000_1 {
    return 100.0;
  }
  let max = ((1 << bit_depth) - 1) as f64;
  20.0 * max.log10() - 10.0 * mse.log10()
}

/// Calculate the mean squared error for a `Plane` by comparing the original (uncompressed)
/// to the compressed version.
fn calculate_plane_mse<T: Pixel>(
  original: &Plane<T>, compressed: &Plane<T>,
) -> f64 {
  original
    .iter()
    .zip(compressed.iter())
    .map(|(a, b)| (i32::cast_from(a) - i32::cast_from(b)).abs() as u64)
    .map(|err| err * err)
    .sum::<u64>() as f64
    / (original.cfg.width * original.cfg.height) as f64
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QualityMetrics {
  /// Peak Signal-to-Noise Ratio for Y, U, and V planes
  pub psnr: Option<PlanarMetrics>,
  /// Peak Signal-to-Noise Ratio as perceived by the Human Visual System--
  /// taking into account Contrast Sensitivity Function (CSF)
  pub psnr_hvs: Option<PlanarMetrics>,
  /// Structural Similarity
  pub ssim: Option<PlanarMetrics>,
  /// Multi-Scale Structural Similarity
  pub ms_ssim: Option<PlanarMetrics>,
  /// CIEDE 2000 color difference algorithm: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
  pub ciede: Option<f64>,
  /// Aligned Peak Signal-to-Noise Ratio for Y, U, and V planes
  pub apsnr: Option<PlanarMetrics>,
  /// Netflix's Video Multimethod Assessment Fusion
  pub vmaf: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricsEnabled {
  /// Don't calculate any metrics.
  None,
  /// Calculate the PSNR of each plane, but no other metrics.
  Psnr,
  /// Calculate all implemented metrics. Currently implemented metrics match what is available via AWCY.
  All,
}

pub fn calculate_frame_metrics<T: Pixel>(
  frame1: &Frame<T>, frame2: &Frame<T>, bit_depth: usize, cs: ChromaSampling,
  metrics: MetricsEnabled,
) -> QualityMetrics {

  #[derive(Clone, Debug)]
  let frame1_info = FrameInfo{
    planes: &frame1.planeData,
    bit_depth: bit_depth,
    chroma_sampling: cs,
  };
  
  let frame2_info = FrameInfo{
    planes: frame2.planes[0],
    bit_depth: bit_depth,
    chroma_sampling: cs,
  };

  match metrics {
    MetricsEnabled::None => QualityMetrics::default(),
    MetricsEnabled::Psnr => {
      let mut metrics = QualityMetrics::default();
      metrics.psnr =
        Some(psnr::calculate_frame_psnr(frame1_info, frame2_info));
      metrics
    }
    MetricsEnabled::All => {
      let mut metrics = QualityMetrics::default();
      metrics.psnr =
        Some(psnr::calculate_frame_psnr(frame1_info, frame2_info));
      metrics.psnr_hvs = Some(psnr_hvs::calculate_frame_psnr_hvs(
        frame1_info, frame2_info
      ));
      let ssim = ssim::calculate_frame_ssim(frame1_info, frame2_info);
      metrics.ssim = Some(ssim);
      let ms_ssim =
        ssim::calculate_frame_msssim(frame1_info, frame2_info);
      metrics.ms_ssim = Some(ms_ssim);
      let ciede = ciede::calculate_frame_ciede(frame1_info, frame2_info);
      metrics.ciede = Some(ciede);
      metrics
    }
  }
}
// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]

use crate::transform::TxSize;
use crate::util::*;

use num_traits::*;
use std::convert::Into;
use std::mem;
use std::ops::AddAssign;

pub trait Coefficient:
  PrimInt
  + Into<i32>
  + AsPrimitive<i32>
  + CastFromPrimitive<i32>
  + AddAssign
  + Signed
  + 'static
{
}
impl Coefficient for i16 {}
impl Coefficient for i32 {}

pub fn get_log_tx_scale(tx_size: TxSize) -> usize {
  let num_pixels = tx_size.area();

  Into::<usize>::into(num_pixels > 256)
    + Into::<usize>::into(num_pixels > 1024)
}

pub fn dc_q(qindex: u8, delta_q: i8, bit_depth: usize) -> i16 {
  let &table = match bit_depth {
    8 => &dc_qlookup_Q3,
    10 => &dc_qlookup_10_Q3,
    12 => &dc_qlookup_12_Q3,
    _ => unimplemented!(),
  };

  table[(qindex as isize + delta_q as isize).max(0).min(255) as usize]
}

pub fn ac_q(qindex: u8, delta_q: i8, bit_depth: usize) -> i16 {
  let &table = match bit_depth {
    8 => &ac_qlookup_Q3,
    10 => &ac_qlookup_10_Q3,
    12 => &ac_qlookup_12_Q3,
    _ => unimplemented!(),
  };

  table[(qindex as isize + delta_q as isize).max(0).min(255) as usize]
}

// TODO: Handle lossless properly.
fn select_qi(quantizer: i64, qlookup: &[i16; QINDEX_RANGE]) -> u8 {
  if quantizer < qlookup[MINQ] as i64 {
    MINQ as u8
  } else if quantizer >= qlookup[MAXQ] as i64 {
    MAXQ as u8
  } else {
    match qlookup.binary_search(&(quantizer as i16)) {
      Ok(qi) => qi as u8,
      Err(qi) => {
        debug_assert!(qi > MINQ);
        debug_assert!(qi <= MAXQ);
        // Pick the closest quantizer in the log domain.
        let qthresh = (qlookup[qi - 1] as i32) * (qlookup[qi] as i32);
        let q2_i32 = (quantizer as i32) * (quantizer as i32);
        if q2_i32 < qthresh {
          (qi - 1) as u8
        } else {
          qi as u8
        }
      }
    }
  }
}

pub fn select_dc_qi(quantizer: i64, bit_depth: usize) -> u8 {
  let qlookup = match bit_depth {
    8 => &dc_qlookup_Q3,
    10 => &dc_qlookup_10_Q3,
    12 => &dc_qlookup_12_Q3,
    _ => unimplemented!(),
  };
  select_qi(quantizer, qlookup)
}

pub fn select_ac_qi(quantizer: i64, bit_depth: usize) -> u8 {
  let qlookup = match bit_depth {
    8 => &ac_qlookup_Q3,
    10 => &ac_qlookup_10_Q3,
    12 => &ac_qlookup_12_Q3,
    _ => unimplemented!(),
  };
  select_qi(quantizer, qlookup)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct QuantizationContext {
  log_tx_scale: usize,
  dc_quant: u32,
  dc_offset: i32,
  dc_mul_add: (u32, u32, u32),

  ac_quant: u32,
  ac_offset_eob: i32,
  ac_offset0: i32,
  ac_offset1: i32,
  ac_mul_add: (u32, u32, u32),
}

fn divu_gen(d: u32) -> (u32, u32, u32) {
  let nbits = (mem::size_of_val(&d) as u64) * 8;
  let m = nbits - d.leading_zeros() as u64 - 1;
  if (d & (d - 1)) == 0 {
    (0xFFFF_FFFF, 0xFFFF_FFFF, m as u32)
  } else {
    let d = d as u64;
    let t = (1u64 << (m + nbits)) / d;
    let r = (t * d + d) & ((1 << nbits) - 1);
    if r <= 1u64 << m {
      (t as u32 + 1, 0u32, m as u32)
    } else {
      (t as u32, t as u32, m as u32)
    }
  }
}

#[inline]
fn divu_pair(x: i32, d: (u32, u32, u32)) -> i32 {
  let y = if x < 0 { -x } else { x } as u64;
  let (a, b, shift) = d;
  let shift = shift as u64;
  let a = a as u64;
  let b = b as u64;

  let y = (((a * y + b) >> 32) >> shift) as i32;
  if x < 0 {
    -y
  } else {
    y
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::transform::TxSize::*;

  #[test]
  fn test_divu_pair() {
    for d in 1..1024 {
      for x in -1000..1000 {
        let ab = divu_gen(d as u32);
        assert_eq!(x / d, divu_pair(x, ab));
      }
    }
  }
  #[test]
  fn gen_divu_table() {
    let b: Vec<(u32, u32, u32)> =
      dc_qlookup_Q3.iter().map(|&v| divu_gen(v as u32)).collect();

    println!("{:?}", b);
  }
  #[test]
  fn test_tx_log_scale() {
    let tx_sizes = [
      (TX_4X4, 0),
      (TX_8X8, 0),
      (TX_16X16, 0),
      (TX_32X32, 1),
      (TX_64X64, 2),
      (TX_4X8, 0),
      (TX_8X4, 0),
      (TX_8X16, 0),
      (TX_16X8, 0),
      (TX_16X32, 1),
      (TX_32X16, 1),
      (TX_32X64, 2),
      (TX_64X32, 2),
      (TX_4X16, 0),
      (TX_16X4, 0),
      (TX_8X32, 0),
      (TX_32X8, 0),
      (TX_16X64, 1),
      (TX_64X16, 1),
    ];
    for &tx_size in tx_sizes.iter() {
      assert!(tx_size.1 == get_log_tx_scale(tx_size.0));
    }
  }
}

impl QuantizationContext {
  pub fn update(
    &mut self, qindex: u8, tx_size: TxSize, is_intra: bool, bit_depth: usize,
    dc_delta_q: i8, ac_delta_q: i8,
  ) {
    self.log_tx_scale = get_log_tx_scale(tx_size);

    self.dc_quant = dc_q(qindex, dc_delta_q, bit_depth) as u32;
    self.dc_mul_add = divu_gen(self.dc_quant);

    self.ac_quant = ac_q(qindex, ac_delta_q, bit_depth) as u32;
    self.ac_mul_add = divu_gen(self.ac_quant);

    // All of these biases were derived by measuring the cost of coding
    // a zero vs coding a one on any given coefficient position, or, in
    // the case of the EOB bias, the cost of coding the block with
    // the chosen EOB (rounding to one) vs rounding to zero and continuing
    // to choose a new EOB. This was done over several clips, with the
    // average of the bit costs taken over all blocks in the set, and a new
    // bias derived via the method outlined in Jean-Marc Valin's
    // Journal of Dubious Theoretical Results[1], aka:
    //
    // lambda = ln(2) / 6.0
    // threshold = 0.5 + (lambda * avg_rate_diff) / 2.0
    // bias = 1 - threshold
    //
    // lambda is a constant since our offsets are already adjusted for the
    // quantizer.
    //
    // Biases were then updated, and cost collection was re-run, until
    // the calculated biases started to converge after 2-4 iterations.
    //
    // In theory, the rounding biases for inter should be somewhat smaller
    // than the biases for intra, but this turns out to only be the case
    // for EOB optimization, or at least, is covered by EOB optimization.
    // The RD-optimal rounding biases for the actual coefficients seem
    // to be quite close (+/- 1/256), for both inter and intra,
    // post-deadzoning.
    //
    // [1] https://people.xiph.org/~jm/notes/theoretical_results.pdf
    self.dc_offset =
      self.dc_quant as i32 * (if is_intra { 109 } else { 108 }) / 256;
    self.ac_offset0 =
      self.ac_quant as i32 * (if is_intra { 98 } else { 97 }) / 256;
    self.ac_offset1 =
      self.ac_quant as i32 * (if is_intra { 109 } else { 108 }) / 256;
    self.ac_offset_eob =
      self.ac_quant as i32 * (if is_intra { 88 } else { 44 }) / 256;
  }

  #[inline]
  pub fn quantize<T>(
    &self, coeffs: &[T], qcoeffs: &mut [T], coded_tx_size: usize,
  ) where
    T: Coefficient,
  {
    // Find the last non-zero coefficient using our smaller biases and
    // zero everything else.
    // This threshold is such that `abs(coeff) < deadzone` implies:
    // (abs(coeff << log_tx_scale) + ac_offset_eob) / ac_quant == 0
    let deadzone = (self.ac_quant as usize - self.ac_offset_eob as usize)
      .align_power_of_two_and_shift(self.log_tx_scale)
      as i32;
    let pos =
      coeffs[1..coded_tx_size].iter().rposition(|c| c.as_().abs() >= deadzone);
    // We skip the DC coefficient since it has its own quantizer index.
    let last_pos = pos.map(|pos| pos + 1).unwrap_or(1);

    qcoeffs[0] = coeffs[0] << (self.log_tx_scale as usize);
    qcoeffs[0] += qcoeffs[0].signum() * T::cast_from(self.dc_offset);
    qcoeffs[0] = T::cast_from(divu_pair(qcoeffs[0].as_(), self.dc_mul_add));

    // Here we use different rounding biases depending on whether we've
    // had recent coefficients that are larger than one, or less than
    // one. The reason for this is that a block usually has a chunk of
    // large coefficients and a tail of zeroes and ones, and the tradeoffs
    // for coding these two are different. In the tail of zeroes and ones,
    // you'll likely end up spending most bits just saying where that
    // coefficient is in the block, whereas in the chunk of larger
    // coefficients, most bits will be spent on coding its magnitude.
    // To that end, we want to bias more toward rounding to zero for
    // that tail of zeroes and ones than we do for the larger coefficients.
    let mut level_mode = 1;
    for (qc, c) in
      qcoeffs[1..].iter_mut().zip(coeffs[1..].iter()).take(last_pos)
    {
      let coeff = *c << self.log_tx_scale;
      let level0 = T::cast_from(divu_pair(coeff.as_(), self.ac_mul_add));
      let offset = if level0 > T::cast_from(1 - level_mode) {
        self.ac_offset1
      } else {
        self.ac_offset0
      };
      let qcoeff = coeff + (coeff.signum() * T::cast_from(offset));
      *qc = T::cast_from(divu_pair(qcoeff.as_(), self.ac_mul_add));
      if level_mode != 0 && *qc == T::cast_from(0) {
        level_mode = 0;
      } else if *qc > T::cast_from(1) {
        level_mode = 1;
      }
    }

    let zero_start = coded_tx_size.min(last_pos + 1);
    if qcoeffs.len() > zero_start {
      for qc in qcoeffs[zero_start..].iter_mut() {
        *qc = T::cast_from(0);
      }
    }
  }
}

pub fn dequantize(
  qindex: u8, coeffs: &[i32], rcoeffs: &mut [i32], tx_size: TxSize,
  bit_depth: usize, dc_delta_q: i8, ac_delta_q: i8,
) {
  let log_tx_scale = get_log_tx_scale(tx_size) as i32;
  let offset = (1 << log_tx_scale) - 1;

  let dc_quant = dc_q(qindex, dc_delta_q, bit_depth) as i32;
  let ac_quant = ac_q(qindex, ac_delta_q, bit_depth) as i32;

  for (i, (r, &c)) in rcoeffs.iter_mut().zip(coeffs.iter()).enumerate() {
    let quant = if i == 0 { dc_quant } else { ac_quant };
    *r = (c * quant + ((c >> 31) & offset)) >> log_tx_scale;
  }
}

// LUTS --------------------------------------------------------------------
const MINQ: usize = 0;
const MAXQ: usize = 255;
const QINDEX_RANGE: usize = MAXQ - MINQ + 1;

#[rustfmt::skip]
static dc_qlookup_Q3: [i16;QINDEX_RANGE] = [
  4,    8,    8,    9,    10,  11,  12,  12,  13,  14,  15,   16,   17,   18,
  19,   19,   20,   21,   22,  23,  24,  25,  26,  26,  27,   28,   29,   30,
  31,   32,   32,   33,   34,  35,  36,  37,  38,  38,  39,   40,   41,   42,
  43,   43,   44,   45,   46,  47,  48,  48,  49,  50,  51,   52,   53,   53,
  54,   55,   56,   57,   57,  58,  59,  60,  61,  62,  62,   63,   64,   65,
  66,   66,   67,   68,   69,  70,  70,  71,  72,  73,  74,   74,   75,   76,
  77,   78,   78,   79,   80,  81,  81,  82,  83,  84,  85,   85,   87,   88,
  90,   92,   93,   95,   96,  98,  99,  101, 102, 104, 105,  107,  108,  110,
  111,  113,  114,  116,  117, 118, 120, 121, 123, 125, 127,  129,  131,  134,
  136,  138,  140,  142,  144, 146, 148, 150, 152, 154, 156,  158,  161,  164,
  166,  169,  172,  174,  177, 180, 182, 185, 187, 190, 192,  195,  199,  202,
  205,  208,  211,  214,  217, 220, 223, 226, 230, 233, 237,  240,  243,  247,
  250,  253,  257,  261,  265, 269, 272, 276, 280, 284, 288,  292,  296,  300,
  304,  309,  313,  317,  322, 326, 330, 335, 340, 344, 349,  354,  359,  364,
  369,  374,  379,  384,  389, 395, 400, 406, 411, 417, 423,  429,  435,  441,
  447,  454,  461,  467,  475, 482, 489, 497, 505, 513, 522,  530,  539,  549,
  559,  569,  579,  590,  602, 614, 626, 640, 654, 668, 684,  700,  717,  736,
  755,  775,  796,  819,  843, 869, 896, 925, 955, 988, 1022, 1058, 1098, 1139,
  1184, 1232, 1282, 1336,
];

#[rustfmt::skip]
static dc_qlookup_10_Q3: [i16;QINDEX_RANGE] = [
  4,    9,    10,   13,   15,   17,   20,   22,   25,   28,   31,   34,   37,
  40,   43,   47,   50,   53,   57,   60,   64,   68,   71,   75,   78,   82,
  86,   90,   93,   97,   101,  105,  109,  113,  116,  120,  124,  128,  132,
  136,  140,  143,  147,  151,  155,  159,  163,  166,  170,  174,  178,  182,
  185,  189,  193,  197,  200,  204,  208,  212,  215,  219,  223,  226,  230,
  233,  237,  241,  244,  248,  251,  255,  259,  262,  266,  269,  273,  276,
  280,  283,  287,  290,  293,  297,  300,  304,  307,  310,  314,  317,  321,
  324,  327,  331,  334,  337,  343,  350,  356,  362,  369,  375,  381,  387,
  394,  400,  406,  412,  418,  424,  430,  436,  442,  448,  454,  460,  466,
  472,  478,  484,  490,  499,  507,  516,  525,  533,  542,  550,  559,  567,
  576,  584,  592,  601,  609,  617,  625,  634,  644,  655,  666,  676,  687,
  698,  708,  718,  729,  739,  749,  759,  770,  782,  795,  807,  819,  831,
  844,  856,  868,  880,  891,  906,  920,  933,  947,  961,  975,  988,  1001,
  1015, 1030, 1045, 1061, 1076, 1090, 1105, 1120, 1137, 1153, 1170, 1186, 1202,
  1218, 1236, 1253, 1271, 1288, 1306, 1323, 1342, 1361, 1379, 1398, 1416, 1436,
  1456, 1476, 1496, 1516, 1537, 1559, 1580, 1601, 1624, 1647, 1670, 1692, 1717,
  1741, 1766, 1791, 1817, 1844, 1871, 1900, 1929, 1958, 1990, 2021, 2054, 2088,
  2123, 2159, 2197, 2236, 2276, 2319, 2363, 2410, 2458, 2508, 2561, 2616, 2675,
  2737, 2802, 2871, 2944, 3020, 3102, 3188, 3280, 3375, 3478, 3586, 3702, 3823,
  3953, 4089, 4236, 4394, 4559, 4737, 4929, 5130, 5347,
];

#[rustfmt::skip]
static dc_qlookup_12_Q3: [i16;QINDEX_RANGE] = [
  4,     12,    18,    25,    33,    41,    50,    60,    70,    80,    91,
  103,   115,   127,   140,   153,   166,   180,   194,   208,   222,   237,
  251,   266,   281,   296,   312,   327,   343,   358,   374,   390,   405,
  421,   437,   453,   469,   484,   500,   516,   532,   548,   564,   580,
  596,   611,   627,   643,   659,   674,   690,   706,   721,   737,   752,
  768,   783,   798,   814,   829,   844,   859,   874,   889,   904,   919,
  934,   949,   964,   978,   993,   1008,  1022,  1037,  1051,  1065,  1080,
  1094,  1108,  1122,  1136,  1151,  1165,  1179,  1192,  1206,  1220,  1234,
  1248,  1261,  1275,  1288,  1302,  1315,  1329,  1342,  1368,  1393,  1419,
  1444,  1469,  1494,  1519,  1544,  1569,  1594,  1618,  1643,  1668,  1692,
  1717,  1741,  1765,  1789,  1814,  1838,  1862,  1885,  1909,  1933,  1957,
  1992,  2027,  2061,  2096,  2130,  2165,  2199,  2233,  2267,  2300,  2334,
  2367,  2400,  2434,  2467,  2499,  2532,  2575,  2618,  2661,  2704,  2746,
  2788,  2830,  2872,  2913,  2954,  2995,  3036,  3076,  3127,  3177,  3226,
  3275,  3324,  3373,  3421,  3469,  3517,  3565,  3621,  3677,  3733,  3788,
  3843,  3897,  3951,  4005,  4058,  4119,  4181,  4241,  4301,  4361,  4420,
  4479,  4546,  4612,  4677,  4742,  4807,  4871,  4942,  5013,  5083,  5153,
  5222,  5291,  5367,  5442,  5517,  5591,  5665,  5745,  5825,  5905,  5984,
  6063,  6149,  6234,  6319,  6404,  6495,  6587,  6678,  6769,  6867,  6966,
  7064,  7163,  7269,  7376,  7483,  7599,  7715,  7832,  7958,  8085,  8214,
  8352,  8492,  8635,  8788,  8945,  9104,  9275,  9450,  9639,  9832,  10031,
  10245, 10465, 10702, 10946, 11210, 11482, 11776, 12081, 12409, 12750, 13118,
  13501, 13913, 14343, 14807, 15290, 15812, 16356, 16943, 17575, 18237, 18949,
  19718, 20521, 21387,
];

#[rustfmt::skip]
static ac_qlookup_Q3: [i16;QINDEX_RANGE] = [
  4,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
  20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,
  33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,
  46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
  59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,
  72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
  85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,
  98,   99,   100,  101,  102,  104,  106,  108,  110,  112,  114,  116,  118,
  120,  122,  124,  126,  128,  130,  132,  134,  136,  138,  140,  142,  144,
  146,  148,  150,  152,  155,  158,  161,  164,  167,  170,  173,  176,  179,
  182,  185,  188,  191,  194,  197,  200,  203,  207,  211,  215,  219,  223,
  227,  231,  235,  239,  243,  247,  251,  255,  260,  265,  270,  275,  280,
  285,  290,  295,  300,  305,  311,  317,  323,  329,  335,  341,  347,  353,
  359,  366,  373,  380,  387,  394,  401,  408,  416,  424,  432,  440,  448,
  456,  465,  474,  483,  492,  501,  510,  520,  530,  540,  550,  560,  571,
  582,  593,  604,  615,  627,  639,  651,  663,  676,  689,  702,  715,  729,
  743,  757,  771,  786,  801,  816,  832,  848,  864,  881,  898,  915,  933,
  951,  969,  988,  1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196,
  1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423, 1451, 1479, 1508, 1537,
  1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
];

#[rustfmt::skip]
static ac_qlookup_10_Q3: [i16;QINDEX_RANGE] = [
  4,    9,    11,   13,   16,   18,   21,   24,   27,   30,   33,   37,   40,
  44,   48,   51,   55,   59,   63,   67,   71,   75,   79,   83,   88,   92,
  96,   100,  105,  109,  114,  118,  122,  127,  131,  136,  140,  145,  149,
  154,  158,  163,  168,  172,  177,  181,  186,  190,  195,  199,  204,  208,
  213,  217,  222,  226,  231,  235,  240,  244,  249,  253,  258,  262,  267,
  271,  275,  280,  284,  289,  293,  297,  302,  306,  311,  315,  319,  324,
  328,  332,  337,  341,  345,  349,  354,  358,  362,  367,  371,  375,  379,
  384,  388,  392,  396,  401,  409,  417,  425,  433,  441,  449,  458,  466,
  474,  482,  490,  498,  506,  514,  523,  531,  539,  547,  555,  563,  571,
  579,  588,  596,  604,  616,  628,  640,  652,  664,  676,  688,  700,  713,
  725,  737,  749,  761,  773,  785,  797,  809,  825,  841,  857,  873,  889,
  905,  922,  938,  954,  970,  986,  1002, 1018, 1038, 1058, 1078, 1098, 1118,
  1138, 1158, 1178, 1198, 1218, 1242, 1266, 1290, 1314, 1338, 1362, 1386, 1411,
  1435, 1463, 1491, 1519, 1547, 1575, 1603, 1631, 1663, 1695, 1727, 1759, 1791,
  1823, 1859, 1895, 1931, 1967, 2003, 2039, 2079, 2119, 2159, 2199, 2239, 2283,
  2327, 2371, 2415, 2459, 2507, 2555, 2603, 2651, 2703, 2755, 2807, 2859, 2915,
  2971, 3027, 3083, 3143, 3203, 3263, 3327, 3391, 3455, 3523, 3591, 3659, 3731,
  3803, 3876, 3952, 4028, 4104, 4184, 4264, 4348, 4432, 4516, 4604, 4692, 4784,
  4876, 4972, 5068, 5168, 5268, 5372, 5476, 5584, 5692, 5804, 5916, 6032, 6148,
  6268, 6388, 6512, 6640, 6768, 6900, 7036, 7172, 7312,
];

#[rustfmt::skip]
static ac_qlookup_12_Q3: [i16;QINDEX_RANGE] = [
  4,     13,    19,    27,    35,    44,    54,    64,    75,    87,    99,
  112,   126,   139,   154,   168,   183,   199,   214,   230,   247,   263,
  280,   297,   314,   331,   349,   366,   384,   402,   420,   438,   456,
  475,   493,   511,   530,   548,   567,   586,   604,   623,   642,   660,
  679,   698,   716,   735,   753,   772,   791,   809,   828,   846,   865,
  884,   902,   920,   939,   957,   976,   994,   1012,  1030,  1049,  1067,
  1085,  1103,  1121,  1139,  1157,  1175,  1193,  1211,  1229,  1246,  1264,
  1282,  1299,  1317,  1335,  1352,  1370,  1387,  1405,  1422,  1440,  1457,
  1474,  1491,  1509,  1526,  1543,  1560,  1577,  1595,  1627,  1660,  1693,
  1725,  1758,  1791,  1824,  1856,  1889,  1922,  1954,  1987,  2020,  2052,
  2085,  2118,  2150,  2183,  2216,  2248,  2281,  2313,  2346,  2378,  2411,
  2459,  2508,  2556,  2605,  2653,  2701,  2750,  2798,  2847,  2895,  2943,
  2992,  3040,  3088,  3137,  3185,  3234,  3298,  3362,  3426,  3491,  3555,
  3619,  3684,  3748,  3812,  3876,  3941,  4005,  4069,  4149,  4230,  4310,
  4390,  4470,  4550,  4631,  4711,  4791,  4871,  4967,  5064,  5160,  5256,
  5352,  5448,  5544,  5641,  5737,  5849,  5961,  6073,  6185,  6297,  6410,
  6522,  6650,  6778,  6906,  7034,  7162,  7290,  7435,  7579,  7723,  7867,
  8011,  8155,  8315,  8475,  8635,  8795,  8956,  9132,  9308,  9484,  9660,
  9836,  10028, 10220, 10412, 10604, 10812, 11020, 11228, 11437, 11661, 11885,
  12109, 12333, 12573, 12813, 13053, 13309, 13565, 13821, 14093, 14365, 14637,
  14925, 15213, 15502, 15806, 16110, 16414, 16734, 17054, 17390, 17726, 18062,
  18414, 18766, 19134, 19502, 19886, 20270, 20670, 21070, 21486, 21902, 22334,
  22766, 23214, 23662, 24126, 24590, 25070, 25551, 26047, 26559, 27071, 27599,
  28143, 28687, 29247,
];
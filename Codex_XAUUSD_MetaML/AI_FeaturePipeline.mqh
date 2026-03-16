#pragma once

// ============================================================================
// AI_FeaturePipeline.mqh
// Verified deployment helper for Python pipeline in trend_following.py
// - Feature set is dynamic (selected per training run)
// - BUY/SELL models use the SAME selected feature vector order
// - RobustScaler is applied with center/scale exported from Python
// ============================================================================

#define FP_MAX_FEATURES 256

struct FP_Config
{
   int    n_features;
   string feature_names[FP_MAX_FEATURES];
   double center[FP_MAX_FEATURES];
   double scale[FP_MAX_FEATURES];

   // Delta-edge decision params from Python best thresholds
   double buy_edge_threshold;
   double sell_edge_threshold;
   double edge_margin;

   // Alignment-critical params (must match Python)
   string entry_mode;        // expected: "next_open"
   int    signal_shift;      // expected: 0
   string barrier_type;      // expected: "atr"
   string same_bar_conflict; // expected: "sl_first"
   int    max_hold;

   // Optional regime adjustment
   bool   use_regime_adjustment;
   double regime_delta;
};

FP_Config g_fp_cfg;

// -----------------------------
// REQUIRED: initialize config from Python export
// -----------------------------
void FP_ResetConfig()
{
   g_fp_cfg.n_features          = 0;
   g_fp_cfg.buy_edge_threshold  = 0.08;
   g_fp_cfg.sell_edge_threshold = 0.08;
   g_fp_cfg.edge_margin         = 0.55;
   g_fp_cfg.entry_mode          = "next_open";
   g_fp_cfg.signal_shift        = 0;
   g_fp_cfg.barrier_type        = "atr";
   g_fp_cfg.same_bar_conflict   = "sl_first";
   g_fp_cfg.max_hold            = 12;
   g_fp_cfg.use_regime_adjustment = false;
   g_fp_cfg.regime_delta        = 0.04;

   for(int i=0;i<FP_MAX_FEATURES;i++)
   {
      g_fp_cfg.feature_names[i] = "";
      g_fp_cfg.center[i] = 0.0;
      g_fp_cfg.scale[i]  = 1.0;
   }
}

bool FP_SetFeatureCount(const int n)
{
   if(n <= 0 || n > FP_MAX_FEATURES)
      return false;
   g_fp_cfg.n_features = n;
   return true;
}

bool FP_SetFeatureDef(const int idx, const string name, const double center, const double scale)
{
   if(idx < 0 || idx >= g_fp_cfg.n_features)
      return false;
   g_fp_cfg.feature_names[idx] = name;
   g_fp_cfg.center[idx] = center;
   g_fp_cfg.scale[idx]  = (MathAbs(scale) < 1e-12 ? 1.0 : scale);
   return true;
}

// -------------------------------------------------------------------------
// NOTE:
// This function computes ONLY verified feature names from features_lib.py.
// If a feature name is not supported, it returns false (no guessing).
// -------------------------------------------------------------------------

double FP_SafeDiv(const double a, const double b, const double fallback=0.0)
{
   if(MathAbs(b) < 1e-12)
      return fallback;
   return a / b;
}

bool FP_LoadRates(MqlRates &rates[], const int min_bars)
{
   int got = CopyRates(_Symbol, PERIOD_H1, 0, min_bars, rates);
   if(got < min_bars)
      return false;
   ArraySetAsSeries(rates, true);
   return true;
}

double FP_RollMeanClose(const MqlRates &r[], const int shift, const int n)
{
   double s=0.0;
   for(int i=shift; i<shift+n; i++) s += r[i].close;
   return s / n;
}

double FP_RollStdClose(const MqlRates &r[], const int shift, const int n)
{
   double m = FP_RollMeanClose(r, shift, n);
   double v = 0.0;
   for(int i=shift; i<shift+n; i++)
   {
      double d = r[i].close - m;
      v += d*d;
   }
   return MathSqrt(v / n);
}

double FP_Highest(const MqlRates &r[], const int shift, const int n)
{
   double hh = -DBL_MAX;
   for(int i=shift; i<shift+n; i++) if(r[i].high > hh) hh = r[i].high;
   return hh;
}

double FP_Lowest(const MqlRates &r[], const int shift, const int n)
{
   double ll = DBL_MAX;
   for(int i=shift; i<shift+n; i++) if(r[i].low < ll) ll = r[i].low;
   return ll;
}

bool FP_ComputeFeature(const string name, const MqlRates &r[], const int shift, double &v)
{
   // -------- Trend features --------
   if(name == "ema_stack_bull" || name == "ema_stack_bear" ||
      name == "bull_regime_score" || name == "bear_regime_score")
   {
      // Approximate EMA by iMA to preserve semantics used in Python
      double ema20  = iMA(_Symbol, PERIOD_H1, 20, 0, MODE_EMA, PRICE_CLOSE, shift);
      double ema50  = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE, shift);
      double ema200 = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE, shift);
      bool bull = (ema20 > ema50 && ema50 > ema200);
      bool bear = (ema20 < ema50 && ema50 < ema200);
      if(name == "ema_stack_bull") { v = (bull ? 1.0 : 0.0); return true; }
      if(name == "ema_stack_bear") { v = (bear ? 1.0 : 0.0); return true; }
      if(name == "bull_regime_score") { v = (bull ? 1.0 : 0.0); return true; }
      if(name == "bear_regime_score") { v = (bear ? 1.0 : 0.0); return true; }
   }

   // -------- Core parsers from features_lib naming --------
   // roc_p
   if(StringSubstr(name,0,4)=="roc_")
   {
      int p = (int)StringToInteger(StringSubstr(name,4));
      v = FP_SafeDiv(r[shift].close, r[shift+p].close, 1.0) - 1.0;
      return true;
   }

   // break_high_p / break_low_p
   if(StringSubstr(name,0,11)=="break_high_")
   {
      int p = (int)StringToInteger(StringSubstr(name,11));
      double hh = FP_Highest(r, shift+1, p);
      v = (r[shift].close > hh ? 1.0 : 0.0);
      return true;
   }
   if(StringSubstr(name,0,10)=="break_low_")
   {
      int p = (int)StringToInteger(StringSubstr(name,10));
      double ll = FP_Lowest(r, shift+1, p);
      v = (r[shift].close < ll ? 1.0 : 0.0);
      return true;
   }

   // dist_to_high_p / dist_to_low_p
   if(StringSubstr(name,0,13)=="dist_to_high_")
   {
      int p = (int)StringToInteger(StringSubstr(name,13));
      double hh = FP_Highest(r, shift, p);
      v = FP_SafeDiv(hh - r[shift].close, r[shift].close, 0.0);
      return true;
   }
   if(StringSubstr(name,0,12)=="dist_to_low_")
   {
      int p = (int)StringToInteger(StringSubstr(name,12));
      double ll = FP_Lowest(r, shift, p);
      v = FP_SafeDiv(r[shift].close - ll, r[shift].close, 0.0);
      return true;
   }

   // Named direct features used frequently
   if(name == "tickvol")                      { v = (double)r[shift].tick_volume; return true; }
   if(name == "atr_14")                       { v = iATR(_Symbol, PERIOD_H1, 14, shift); return true; }
   if(name == "atr_14_pct_close")             { v = FP_SafeDiv(iATR(_Symbol, PERIOD_H1, 14, shift), r[shift].close, 0.0); return true; }
   if(name == "body_ratio")
   {
      double cr = r[shift].high - r[shift].low;
      v = FP_SafeDiv(MathAbs(r[shift].close - r[shift].open), cr, 0.0);
      return true;
   }
   if(name == "bull_body_flag")               { v = (r[shift].close > r[shift].open ? 1.0 : 0.0); return true; }
   if(name == "bear_body_flag")               { v = (r[shift].close < r[shift].open ? 1.0 : 0.0); return true; }

   // NOT SUPPORTED => no guessing
   return false;
}

bool FP_BuildScaledFeatures(const int shift, double &x[])
{
   if(g_fp_cfg.n_features <= 0)
      return false;

   MqlRates rates[];
   if(!FP_LoadRates(rates, 800))
      return false;

   ArrayResize(x, g_fp_cfg.n_features);
   for(int i=0; i<g_fp_cfg.n_features; i++)
   {
      string fname = g_fp_cfg.feature_names[i];
      double raw = 0.0;
      if(!FP_ComputeFeature(fname, rates, shift, raw))
      {
         Print("Unsupported feature for strict parity: ", fname);
         return false;
      }

      double den = g_fp_cfg.scale[i];
      if(MathAbs(den) < 1e-12) den = 1.0;
      x[i] = (raw - g_fp_cfg.center[i]) / den;
      if(!MathIsValidNumber(x[i]))
         return false;
   }
   return true;
}

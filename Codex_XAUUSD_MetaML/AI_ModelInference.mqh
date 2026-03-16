#pragma once

#include "AI_FeaturePipeline.mqh"

// ============================================================================
// AI_ModelInference.mqh
// Verified runtime behavior from Python:
// - two binary models: BUY edge and SELL edge
// - both read SAME scaled feature vector
// - model output used is probability of class-1 (predict_proba[:,1])
// - final action uses delta-edge thresholds
// ============================================================================

// ONNX paths (set from EA inputs or hardcode)
string g_buy_model_path  = "";
string g_sell_model_path = "";

struct MI_Context
{
   long buy_model;
   long sell_model;
   bool ready;
};

bool MI_Load(const string buy_path, const string sell_path, MI_Context &ctx)
{
   ctx.buy_model = INVALID_HANDLE;
   ctx.sell_model = INVALID_HANDLE;
   ctx.ready = false;

   ctx.buy_model = OnnxCreate(buy_path);
   if(ctx.buy_model == INVALID_HANDLE)
   {
      Print("OnnxCreate buy failed: ", buy_path, " err=", GetLastError());
      return false;
   }

   ctx.sell_model = OnnxCreate(sell_path);
   if(ctx.sell_model == INVALID_HANDLE)
   {
      Print("OnnxCreate sell failed: ", sell_path, " err=", GetLastError());
      OnnxRelease(ctx.buy_model);
      ctx.buy_model = INVALID_HANDLE;
      return false;
   }

   ctx.ready = true;
   return true;
}

void MI_Close(MI_Context &ctx)
{
   if(ctx.buy_model != INVALID_HANDLE) OnnxRelease(ctx.buy_model);
   if(ctx.sell_model != INVALID_HANDLE) OnnxRelease(ctx.sell_model);
   ctx.buy_model = INVALID_HANDLE;
   ctx.sell_model = INVALID_HANDLE;
   ctx.ready = false;
}

bool MI_RunProbClass1(const long handle, const double &x[], double &prob1)
{
   if(handle == INVALID_HANDLE)
      return false;

   vectorf in(ArraySize(x));
   for(int i=0; i<ArraySize(x); i++) in[i] = (float)x[i];

   // Default assumption from CatBoost ONNX export in this repo:
   // second output contains class probabilities and class-1 at index 1.
   vectorf probs(2);
   if(!OnnxRun(handle, ONNX_NO_CONVERSION, in, probs))
   {
      Print("OnnxRun failed err=", GetLastError());
      return false;
   }

   prob1 = (double)probs[1];
   if(!MathIsValidNumber(prob1)) return false;
   if(prob1 < 0.0) prob1 = 0.0;
   if(prob1 > 1.0) prob1 = 1.0;
   return true;
}

// BUY=0, SELL=1, NO_TRADE=-1
int MI_DecideAction(const double prob_buy, const double prob_sell,
                    const double buy_edge_threshold,
                    const double sell_edge_threshold,
                    const double edge_margin)
{
   double delta = prob_buy - prob_sell;
   bool buy_signal  = (delta >  buy_edge_threshold) && (prob_buy  >= edge_margin);
   bool sell_signal = (delta < -sell_edge_threshold) && (prob_sell >= edge_margin);

   if(buy_signal && !sell_signal)  return 0;
   if(sell_signal && !buy_signal)  return 1;
   return -1;
}

bool MI_Infer(const MI_Context &ctx,
              const int feature_shift,
              double &prob_buy,
              double &prob_sell,
              double &delta,
              int &action)
{
   if(!ctx.ready)
      return false;

   // Python alignment expects signal_shift=0 and entry on next_open.
   // In MT5 we therefore compute signal on closed bar (shift=1 by caller)
   // and let EA execute on next bar open.
   double x[];
   if(!FP_BuildScaledFeatures(feature_shift, x))
      return false;

   if(!MI_RunProbClass1(ctx.buy_model, x, prob_buy))
      return false;
   if(!MI_RunProbClass1(ctx.sell_model, x, prob_sell))
      return false;

   delta = prob_buy - prob_sell;
   action = MI_DecideAction(prob_buy, prob_sell,
                            g_fp_cfg.buy_edge_threshold,
                            g_fp_cfg.sell_edge_threshold,
                            g_fp_cfg.edge_margin);
   return true;
}

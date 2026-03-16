#pragma once

#include <Trade/Trade.mqh>
#include "AI_FeaturePipeline.mqh"

// ============================================================
// AI_ModelInference.mqh
// ONNX model loading + inference wrappers for buy/sell models.
// ============================================================

// Replace with your exported model filenames
input string InpBuyModelPath  = "XAUUSD_H1_buy_model.onnx";
input string InpSellModelPath = "XAUUSD_H1_sell_model.onnx";

struct AIInferenceContext
{
   long buy_handle;
   long sell_handle;
   bool ready;
};

bool AI_LoadModel(const string path, long &handle)
{
   handle = OnnxCreate(path);
   if(handle == INVALID_HANDLE)
   {
      Print("OnnxCreate failed for path: ", path, " | err=", GetLastError());
      return false;
   }
   return true;
}

bool AI_Init(AIInferenceContext &ctx)
{
   ctx.ready = false;
   ctx.buy_handle = INVALID_HANDLE;
   ctx.sell_handle = INVALID_HANDLE;

   if(!AI_LoadModel(InpBuyModelPath, ctx.buy_handle))
      return false;

   if(!AI_LoadModel(InpSellModelPath, ctx.sell_handle))
   {
      OnnxRelease(ctx.buy_handle);
      ctx.buy_handle = INVALID_HANDLE;
      return false;
   }

   ctx.ready = true;
   return true;
}

void AI_Shutdown(AIInferenceContext &ctx)
{
   if(ctx.buy_handle != INVALID_HANDLE)
      OnnxRelease(ctx.buy_handle);
   if(ctx.sell_handle != INVALID_HANDLE)
      OnnxRelease(ctx.sell_handle);

   ctx.buy_handle = INVALID_HANDLE;
   ctx.sell_handle = INVALID_HANDLE;
   ctx.ready = false;
}

// Run one model and return probability of class-1
bool AI_RunSingle(long handle, const double &feat[], double &prob_out)
{
   if(handle == INVALID_HANDLE)
      return false;

   vectorf x(N_FEATURES);
   for(int i=0; i<N_FEATURES; i++)
      x[i] = (float)feat[i];

   // Most sklearn/catboost ONNX binary classifiers expose
   // probabilities [N,2] or [2]. We'll read 2-prob output.
   vectorf probs(2);

   // IMPORTANT: input/output names can differ by exporter.
   // Update names if needed using your ONNX graph inspection.
   if(!OnnxRun(handle, ONNX_NO_CONVERSION, x, probs))
   {
      Print("OnnxRun failed. err=", GetLastError());
      return false;
   }

   prob_out = (double)probs[1];
   if(!MathIsValidNumber(prob_out))
      return false;

   // Clamp to [0,1]
   if(prob_out < 0.0) prob_out = 0.0;
   if(prob_out > 1.0) prob_out = 1.0;

   return true;
}

bool AI_Predict(AIInferenceContext &ctx,
                const double &scaled_feat[],
                double &prob_buy,
                double &prob_sell,
                double &delta)
{
   if(!ctx.ready)
      return false;

   if(ArraySize(scaled_feat) != N_FEATURES)
      return false;

   if(!AI_RunSingle(ctx.buy_handle, scaled_feat, prob_buy))
      return false;

   if(!AI_RunSingle(ctx.sell_handle, scaled_feat, prob_sell))
      return false;

   delta = prob_buy - prob_sell;
   return MathIsValidNumber(delta);
}

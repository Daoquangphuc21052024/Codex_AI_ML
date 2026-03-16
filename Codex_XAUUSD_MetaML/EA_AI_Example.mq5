#property strict

#include <Trade/Trade.mqh>
#include "AI_ModelInference.mqh"

// ============================================================================
// EA_AI_Example.mq5
// Safe deployment harness:
// - Default: inference-only logging (no trades)
// - Runs on H1 closed bar to match Python signal timing
// ============================================================================

input bool   InpDryRunOnly = true;
input string InpBuyModelPath  = "XAUUSD_H1_model_0.onnx";
input string InpSellModelPath = "XAUUSD_H1_meta_0.onnx";

input double InpLot = 0.10;
input int    InpSLPoints = 600;
input int    InpTPPoints = 600;

CTrade      g_trade;
MI_Context  g_ctx;
datetime    g_last_closed_bar = 0;

bool EA_IsNewClosedH1Bar()
{
   MqlRates r[];
   if(CopyRates(_Symbol, PERIOD_H1, 0, 3, r) < 3)
      return false;
   ArraySetAsSeries(r, true);
   datetime t = r[1].time;
   if(t != g_last_closed_bar)
   {
      g_last_closed_bar = t;
      return true;
   }
   return false;
}

bool EA_HasOpenPos()
{
   return PositionSelect(_Symbol);
}

bool EA_Send(const int action)
{
   if(action == -1)
      return true;

   if(EA_HasOpenPos())
   {
      Print("Skip order: existing position on symbol");
      return true;
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(action == 0)
   {
      double sl = ask - InpSLPoints * _Point;
      double tp = ask + InpTPPoints * _Point;
      return g_trade.Buy(InpLot, _Symbol, ask, sl, tp, "AI_BUY");
   }
   else
   {
      double sl = bid + InpSLPoints * _Point;
      double tp = bid - InpTPPoints * _Point;
      return g_trade.Sell(InpLot, _Symbol, bid, sl, tp, "AI_SELL");
   }
}

int OnInit()
{
   FP_ResetConfig();

   // -----------------------------------------------------------------
   // REQUIRED: paste exact export config from Python report json
   // Example only (replace with real selected features + scaler values)
   // -----------------------------------------------------------------
   FP_SetFeatureCount(4);
   FP_SetFeatureDef(0, "roc_3", 0.0, 1.0);
   FP_SetFeatureDef(1, "atr_14", 0.0, 1.0);
   FP_SetFeatureDef(2, "bull_regime_score", 0.0, 1.0);
   FP_SetFeatureDef(3, "bear_regime_score", 0.0, 1.0);

   g_fp_cfg.buy_edge_threshold  = 0.08;
   g_fp_cfg.sell_edge_threshold = 0.08;
   g_fp_cfg.edge_margin         = 0.55;
   g_fp_cfg.entry_mode          = "next_open";
   g_fp_cfg.signal_shift        = 0;
   g_fp_cfg.barrier_type        = "atr";
   g_fp_cfg.same_bar_conflict   = "sl_first";
   g_fp_cfg.max_hold            = 12;

   if(!MI_Load(InpBuyModelPath, InpSellModelPath, g_ctx))
      return INIT_FAILED;

   Print("EA initialized. DryRunOnly=", InpDryRunOnly);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   MI_Close(g_ctx);
}

void OnTick()
{
   if(!EA_IsNewClosedH1Bar())
      return;

   double prob_buy=0.0, prob_sell=0.0, delta=0.0;
   int action=-1;

   // shift=1 => last closed bar, matching Python decision timing
   if(!MI_Infer(g_ctx, 1, prob_buy, prob_sell, delta, action))
   {
      Print("MI_Infer failed. Check feature parity / scaler / model outputs.");
      return;
   }

   string act = (action==0?"BUY":(action==1?"SELL":"NO_TRADE"));
   PrintFormat("AI | buy=%.6f sell=%.6f delta=%.6f action=%s", prob_buy, prob_sell, delta, act);

   if(InpDryRunOnly)
      return;

   if(!EA_Send(action))
      Print("Order failed err=", GetLastError());
}

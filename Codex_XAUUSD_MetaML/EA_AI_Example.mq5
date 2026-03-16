#property strict

#include <Trade/Trade.mqh>
#include "AI_ModelInference.mqh"

// ============================================================
// EA_AI_Example.mq5
// Example EA:
// - computes features on closed H1 bar
// - runs ONNX buy/sell models
// - uses delta-edge action model
// - can run in SAFE mode (no orders)
// ============================================================

input bool   InpSafeInferenceOnly = true;
input double InpLotSize           = 0.10;
input int    InpSLPoints          = 600;
input int    InpTPPoints          = 600;

CTrade g_trade;
AIInferenceContext g_ai;
datetime g_lastBarTime = 0;

bool IsNewClosedBar()
{
   MqlRates rates[];
   int got = CopyRates(_Symbol, PERIOD_H1, 0, 3, rates);
   if(got < 3)
      return false;

   ArraySetAsSeries(rates, true);
   datetime lastClosedBar = rates[1].time;
   if(lastClosedBar != g_lastBarTime)
   {
      g_lastBarTime = lastClosedBar;
      return true;
   }
   return false;
}

bool HasPositionOnSymbol()
{
   if(!PositionSelect(_Symbol))
      return false;
   return true;
}

void LogDecision(const double prob_buy,
                 const double prob_sell,
                 const double delta,
                 const int action)
{
   string a = (action == 0 ? "BUY" : (action == 1 ? "SELL" : "NO_TRADE"));
   PrintFormat("AI decision | prob_buy=%.6f prob_sell=%.6f delta=%.6f action=%s",
               prob_buy, prob_sell, delta, a);
}

bool PlaceOrder(const int action)
{
   if(action == -1)
      return true;

   if(HasPositionOnSymbol())
   {
      Print("Skip: position already exists on symbol");
      return true;
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(action == 0)
   {
      double sl = ask - InpSLPoints * _Point;
      double tp = ask + InpTPPoints * _Point;
      return g_trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "AI BUY");
   }
   else
   {
      double sl = bid + InpSLPoints * _Point;
      double tp = bid - InpTPPoints * _Point;
      return g_trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "AI SELL");
   }
}

int OnInit()
{
   if(!AI_Init(g_ai))
   {
      Print("AI init failed");
      return(INIT_FAILED);
   }

   Print("EA_AI_Example initialized");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   AI_Shutdown(g_ai);
}

void OnTick()
{
   if(!IsNewClosedBar())
      return;

   // shift=1 => last fully closed bar
   double feat[];
   if(!FP_BuildModelInput(1, feat))
   {
      Print("Feature pipeline failed (insufficient bars / invalid feature)");
      return;
   }

   double prob_buy = 0.0, prob_sell = 0.0, delta = 0.0;
   if(!AI_Predict(g_ai, feat, prob_buy, prob_sell, delta))
   {
      Print("Inference failed");
      return;
   }

   double buy_thr = InpBuyEdgeThreshold;
   double sell_thr = InpSellEdgeThreshold;

   // Optional regime adjust (must match Python config)
   if(InpUseRegimeAdjust)
   {
      // proxy regime from features[10]=bull_score, [11]=bear_score in this template
      double bull = feat[10];
      double bear = feat[11];
      buy_thr  = MathMax(0.02, MathMin(0.60, buy_thr  + InpRegimeDelta * (bear - bull)));
      sell_thr = MathMax(0.02, MathMin(0.60, sell_thr + InpRegimeDelta * (bull - bear)));
   }

   int action = FP_DecideAction(prob_buy, prob_sell, buy_thr, sell_thr, InpEdgeMargin);
   LogDecision(prob_buy, prob_sell, delta, action);

   if(InpSafeInferenceOnly)
      return;

   if(!PlaceOrder(action))
      Print("Order send failed. err=", GetLastError());
}

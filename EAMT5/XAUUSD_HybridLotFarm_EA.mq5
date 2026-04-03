#property strict
#property version   "1.00"
#property description "Hybrid Scalping Basket + Soft Grid + Risk Governor EA for XAUUSD lot farming"

#include "LotFarm/Types.mqh"
#include "LotFarm/Inputs.mqh"
#include "LotFarm/Core.mqh"

int OnInit()
{
   if(_Symbol != "XAUUSD" && _Symbol != "XAUUSDm")
      Print("Warning: EA designed for XAUUSD. Current symbol: ", _Symbol);

   if(!InitIndicators())
   {
      Print("Failed to initialize indicators.");
      return(INIT_FAILED);
   }

   g_buyBasket.Reset(+1, _Symbol);
   g_sellBasket.Reset(-1, _Symbol);
   g_daily.Reset(DateKey(TimeCurrent()));
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

   Print("XAUUSD_HybridLotFarm_EA initialized.");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   ReleaseIndicators();
   Comment("");
   Print("XAUUSD_HybridLotFarm_EA deinitialized. reason=", reason);
}

void OnTick()
{
   ProcessTick();
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   OnDealUpdate(trans);
}

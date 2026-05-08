#property strict
#property version   "1.10"
#property description "Hybrid Scalping Basket + Soft Grid + Risk Governor EA for XAUUSD lot farming"

#include <Trade/Trade.mqh>

// =========================================================
// ENUMS / STRUCTS
// =========================================================
enum EntryMode
{
   ENTRY_MEAN_REVERSION = 0,
   ENTRY_BREAKOUT       = 1,
   ENTRY_HYBRID         = 2
};

enum GridMode
{
   GRID_FIXED       = 0,
   GRID_ADDITIVE    = 1,
   GRID_SOFT_MULT   = 2
};

struct BasketInfo
{
   int      basketId;
   int      direction;            // +1 buy, -1 sell
   string   symbol;
   double   totalLots;
   double   weightedAvgPrice;
   double   floatingProfit;
   int      numberOfOrders;
   datetime creationTime;
   datetime lastAddTime;
   double   lastAddPrice;
   bool     isActive;
   double   peakProfit;
   bool     trailingArmed;

   void Reset(const int dir, const string sym)
   {
      basketId          = 0;
      direction         = dir;
      symbol            = sym;
      totalLots         = 0.0;
      weightedAvgPrice  = 0.0;
      floatingProfit    = 0.0;
      numberOfOrders    = 0;
      creationTime      = 0;
      lastAddTime       = 0;
      lastAddPrice      = 0.0;
      isActive          = false;
      peakProfit        = 0.0;
      trailingArmed     = false;
   }
};

struct DailyStats
{
   string   todayDate;
   double   openedLots;
   double   closedLots;
   int      totalTrades;
   int      totalBasketsClosed;
   double   dailyPnL;
   double   maxFloatingDD;

   void Reset(const string dateKey)
   {
      todayDate           = dateKey;
      openedLots          = 0.0;
      closedLots          = 0.0;
      totalTrades         = 0;
      totalBasketsClosed  = 0;
      dailyPnL            = 0.0;
      maxFloatingDD       = 0.0;
   }
};

struct MarketState
{
   double spread;
   double atr;
   double trendSlope;
   double distanceFromMean;
   int    volatilityState; // 0 normal, 1 elevated, 2 panic
   bool   isPanic;
   bool   isTrending;
   bool   isRangeFriendly;
};

// =========================================================
// INPUTS
// =========================================================
input group "[A] General"
input long   InpMagicNumber             = 260401;
input bool   InpEnableBuy               = true;
input bool   InpEnableSell              = true;
input double InpMaxSpread               = 80.0; // points
input bool   InpUseTradingHours         = true;
input int    InpStartHour               = 7;
input int    InpEndHour                 = 22;
input int    InpSlippagePoints          = 30;
input bool   InpDebugMode               = true;

input group "[B] Entry Settings"
input EntryMode InpEntryMode            = ENTRY_HYBRID;
input int    InpRSI_Period              = 8;
input double InpRSI_OB                  = 72.0;
input double InpRSI_OS                  = 28.0;
input int    InpEMA_Period              = 55;
input int    InpBollinger_Period        = 20;
input double InpBollinger_Deviation     = 2.0;
input int    InpBreakoutLookback        = 12;

input group "[C] Grid Settings"
input double   InpBaseLot               = 0.02;
input double   InpGridStepPoints        = 180.0;
input int      InpMaxGridOrdersPerSide  = 6;
input GridMode InpGridMode              = GRID_SOFT_MULT;
input double   InpSoftMultiplier        = 1.18;
input double   InpAdditiveLotStep       = 0.01;
input double   InpMaxLotPerOrder        = 0.20;
input double   InpMaxTotalExposureLots  = 2.00;

input group "[D] Basket Close Settings"
input double InpBasketTakeProfitMoney   = 8.0;
input bool   InpBasketTrailingEnabled   = true;
input double InpBasketTrailingStart     = 10.0;
input double InpBasketTrailingStep      = 2.0;
input int    InpBasketMaxLifetimeMinutes = 240;

input group "[E] Risk Filters"
input int    InpATR_Period              = 14;
input double InpATR_DangerThreshold     = 7.5;
input double InpTrendSlopeThreshold     = 2.0;
input double InpDistanceDangerThreshold = 10.0;
input double InpPanicCandleMultiplier   = 2.2;

input group "[F] Daily Governor"
input double InpDailyTargetClosedLots   = 8.0;
input double InpDailyMaxOpenedLots      = 16.0;
input int    InpDailyMaxTrades          = 280;
input bool   InpDailyLightModeAfterTarget = true;

input group "[G] Kill Switch"
input double InpDailyEquityStopPercent  = 6.0;
input double InpMaxBasketLossMoney      = 70.0;
input double InpMaxFloatingDDPercent    = 10.0;
input int    InpMaxOpenOrders           = 20;
input bool   InpEnablePanicModeStop     = true;

// =========================================================
// GLOBALS
// =========================================================
CTrade g_trade;

MarketState g_market;
DailyStats  g_daily;
BasketInfo  g_buyBasket;
BasketInfo  g_sellBasket;

datetime g_lastBarTime = 0;
double   g_dayStartEquity = 0.0;
bool     g_tradingHaltedToday = false;
bool     g_disableNewEntries  = false;
bool     g_lightMode          = false;

int g_rsiHandle = INVALID_HANDLE;
int g_emaHandle = INVALID_HANDLE;
int g_bbHandle  = INVALID_HANDLE;
int g_atrHandle = INVALID_HANDLE;

// =========================================================
// HELPERS / ENGINES
// =========================================================
string DateKey(datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return StringFormat("%04d-%02d-%02d", dt.year, dt.mon, dt.day);
}

bool IsNewBar()
{
   datetime currentBar = iTime(_Symbol, _Period, 0);
   if(currentBar <= 0)
      return false;

   if(currentBar != g_lastBarTime)
   {
      g_lastBarTime = currentBar;
      return true;
   }
   return false;
}

double NormalizeLots(const double lots)
{
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   double clamped = MathMax(minLot, MathMin(maxLot, lots));
   clamped = MathFloor(clamped / lotStep) * lotStep;
   return NormalizeDouble(clamped, 2);
}

bool CheckSpreadOK()
{
   return g_market.spread <= InpMaxSpread;
}

bool CheckSessionOK()
{
   if(!InpUseTradingHours)
      return true;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(InpStartHour <= InpEndHour)
      return (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   return (dt.hour >= InpStartHour || dt.hour < InpEndHour);
}

bool CheckATRDanger()
{
   return g_market.atr >= InpATR_DangerThreshold;
}

bool CheckTrendDanger()
{
   return MathAbs(g_market.trendSlope) >= InpTrendSlopeThreshold;
}

bool CheckDistanceDanger()
{
   return MathAbs(g_market.distanceFromMean) >= InpDistanceDangerThreshold;
}

bool CheckPanicMode()
{
   double high1 = iHigh(_Symbol, _Period, 1);
   double low1  = iLow(_Symbol, _Period, 1);
   double rangePoints = (high1 - low1) / _Point;
   bool candlePanic = (g_market.atr > 0.0 && rangePoints >= g_market.atr * InpPanicCandleMultiplier);

   bool panic = (CheckATRDanger() && CheckTrendDanger()) || !CheckSpreadOK() || candlePanic;
   return panic;
}

void UpdateMarketState()
{
   MqlTick tick;
   SymbolInfoTick(_Symbol, tick);
   g_market.spread = (tick.ask - tick.bid) / _Point;

   double atrBuffer[];
   if(CopyBuffer(g_atrHandle, 0, 0, 2, atrBuffer) == 2)
      g_market.atr = atrBuffer[0] / _Point;

   double emaBuffer[];
   if(CopyBuffer(g_emaHandle, 0, 0, 3, emaBuffer) == 3)
   {
      g_market.trendSlope = (emaBuffer[0] - emaBuffer[2]) / _Point;
      double mid = emaBuffer[0];
      g_market.distanceFromMean = (tick.bid - mid) / _Point;
   }

   g_market.isTrending = CheckTrendDanger();
   g_market.isPanic = CheckPanicMode();
   g_market.isRangeFriendly = (!g_market.isTrending && !g_market.isPanic);
   g_market.volatilityState = g_market.isPanic ? 2 : (CheckATRDanger() ? 1 : 0);
}

bool CheckMeanReversionBuySignal()
{
   if(!g_market.isRangeFriendly)
      return false;

   double rsi[];
   double bbLower[];
   double bbMid[];
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyBuffer(g_rsiHandle, 0, 1, 1, rsi) != 1) return false;
   if(CopyBuffer(g_bbHandle, 2, 1, 1, bbLower) != 1) return false;
   if(CopyBuffer(g_bbHandle, 0, 1, 1, bbMid) != 1) return false;
   if(CopyRates(_Symbol, _Period, 1, 2, rates) != 2) return false;

   bool oversold = rsi[0] <= InpRSI_OS;
   bool touchBand = rates[0].low <= bbLower[0];
   bool rejection = rates[0].close > rates[0].open;
   bool stretched = rates[0].close < bbMid[0];

   return oversold && touchBand && rejection && stretched;
}

bool CheckMeanReversionSellSignal()
{
   if(!g_market.isRangeFriendly)
      return false;

   double rsi[];
   double bbUpper[];
   double bbMid[];
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyBuffer(g_rsiHandle, 0, 1, 1, rsi) != 1) return false;
   if(CopyBuffer(g_bbHandle, 1, 1, 1, bbUpper) != 1) return false;
   if(CopyBuffer(g_bbHandle, 0, 1, 1, bbMid) != 1) return false;
   if(CopyRates(_Symbol, _Period, 1, 2, rates) != 2) return false;

   bool overbought = rsi[0] >= InpRSI_OB;
   bool touchBand = rates[0].high >= bbUpper[0];
   bool rejection = rates[0].close < rates[0].open;
   bool stretched = rates[0].close > bbMid[0];

   return overbought && touchBand && rejection && stretched;
}

bool CheckMicroBreakoutBuySignal()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int n = InpBreakoutLookback + 3;
   if(CopyRates(_Symbol, _Period, 0, n, rates) != n)
      return false;

   double highest = rates[2].high;
   for(int i = 2; i < n; i++)
      highest = MathMax(highest, rates[i].high);

   bool breakout = rates[1].close > highest;
   bool volExpand = ((rates[1].high - rates[1].low) / _Point) > (g_market.atr * 0.7);
   return breakout && volExpand;
}

bool CheckMicroBreakoutSellSignal()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int n = InpBreakoutLookback + 3;
   if(CopyRates(_Symbol, _Period, 0, n, rates) != n)
      return false;

   double lowest = rates[2].low;
   for(int i = 2; i < n; i++)
      lowest = MathMin(lowest, rates[i].low);

   bool breakout = rates[1].close < lowest;
   bool volExpand = ((rates[1].high - rates[1].low) / _Point) > (g_market.atr * 0.7);
   return breakout && volExpand;
}

int CountOpenOrders()
{
   int total = 0;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      total++;
   }
   return total;
}

double GetTotalOpenLots()
{
   double lots = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      lots += PositionGetDouble(POSITION_VOLUME);
   }
   return lots;
}

bool CheckDailyVolumeCap()
{
   if(g_daily.openedLots >= InpDailyMaxOpenedLots) return false;
   if(g_daily.closedLots >= InpDailyTargetClosedLots && !InpDailyLightModeAfterTarget) return false;
   return true;
}

bool CheckDailyTradeCap()
{
   return g_daily.totalTrades < InpDailyMaxTrades;
}

bool CheckDailyLossCap()
{
   if(g_dayStartEquity <= 0.0) return true;
   double ddPct = (g_dayStartEquity - AccountInfoDouble(ACCOUNT_EQUITY)) / g_dayStartEquity * 100.0;
   return ddPct < InpDailyEquityStopPercent;
}

bool CanOpenNewTrade()
{
   if(g_tradingHaltedToday || g_disableNewEntries) return false;
   if(!CheckSpreadOK() || !CheckSessionOK()) return false;
   if(!CheckDailyVolumeCap() || !CheckDailyTradeCap() || !CheckDailyLossCap()) return false;
   if(CountOpenOrders() >= InpMaxOpenOrders) return false;
   if(GetTotalOpenLots() >= InpMaxTotalExposureLots) return false;
   if(g_market.isPanic && InpEnablePanicModeStop) return false;
   return true;
}

double CalculateNextLot(int direction, int basketDepth)
{
   double lot = InpBaseLot;

   if(g_lightMode)
      lot *= 0.5;

   if(InpGridMode == GRID_ADDITIVE)
      lot = InpBaseLot + (basketDepth * InpAdditiveLotStep);
   else if(InpGridMode == GRID_SOFT_MULT)
      lot = InpBaseLot * MathPow(InpSoftMultiplier, basketDepth);

   lot = MathMin(lot, InpMaxLotPerOrder);

   double remaining = InpMaxTotalExposureLots - GetTotalOpenLots();
   lot = MathMin(lot, remaining);

   return NormalizeLots(lot);
}

int BuildBasketId(const int direction)
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return dt.year * 100000 + dt.mon * 1000 + dt.day * 10 + (direction > 0 ? 1 : 2);
}

bool ExecuteOrder(const int direction, const double lot, const string comment)
{
   if(lot <= 0.0)
      return false;

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpSlippagePoints);

   bool ok = false;
   if(direction > 0)
      ok = g_trade.Buy(lot, _Symbol, 0.0, 0.0, 0.0, comment);
   else
      ok = g_trade.Sell(lot, _Symbol, 0.0, 0.0, 0.0, comment);

   if(!ok && InpDebugMode)
      Print("Order failed: ", comment, " rc=", g_trade.ResultRetcode(), " ", g_trade.ResultRetcodeDescription());

   return ok;
}

void OpenInitialTrade(int direction)
{
   double lot = CalculateNextLot(direction, 0);
   if(lot <= 0.0) return;

   int basketId = BuildBasketId(direction);
   string comment = StringFormat("LF_INIT_%d", basketId);

   if(ExecuteOrder(direction, lot, comment))
   {
      g_daily.openedLots += lot;
      g_daily.totalTrades++;
      if(InpDebugMode) Print("Initial trade opened dir=", direction, " lot=", lot, " basket=", basketId);
   }
}

void OpenGridTrade(int direction, int basketId)
{
   int basketDepth = (direction > 0 ? g_buyBasket.numberOfOrders : g_sellBasket.numberOfOrders);
   double lot = CalculateNextLot(direction, basketDepth);
   if(lot <= 0.0) return;

   string comment = StringFormat("LF_GRID_%d", basketId);
   if(ExecuteOrder(direction, lot, comment))
   {
      g_daily.openedLots += lot;
      g_daily.totalTrades++;
      if(direction > 0)
         g_buyBasket.lastAddTime = TimeCurrent();
      else
         g_sellBasket.lastAddTime = TimeCurrent();
      if(InpDebugMode) Print("Grid trade opened dir=", direction, " lot=", lot, " basket=", basketId);
   }
}

void UpdateBasketStats()
{
   g_buyBasket.Reset(+1, _Symbol);
   g_sellBasket.Reset(-1, _Symbol);

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      double vol = PositionGetDouble(POSITION_VOLUME);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      int type = (int)PositionGetInteger(POSITION_TYPE);
      if(type == POSITION_TYPE_BUY)
      {
         g_buyBasket.weightedAvgPrice = ((g_buyBasket.weightedAvgPrice * g_buyBasket.totalLots) + (openPrice * vol)) / (g_buyBasket.totalLots + vol);
         g_buyBasket.totalLots += vol;
         g_buyBasket.floatingProfit += PositionGetDouble(POSITION_PROFIT);
         g_buyBasket.numberOfOrders++;
         g_buyBasket.isActive = true;
         if(g_buyBasket.basketId == 0)
            g_buyBasket.basketId = BuildBasketId(g_buyBasket.direction);
         if(g_buyBasket.creationTime == 0)
            g_buyBasket.creationTime = (datetime)PositionGetInteger(POSITION_TIME);

         if(openPrice > g_buyBasket.lastAddPrice)
            g_buyBasket.lastAddPrice = openPrice;
      }
      else if(type == POSITION_TYPE_SELL)
      {
         g_sellBasket.weightedAvgPrice = ((g_sellBasket.weightedAvgPrice * g_sellBasket.totalLots) + (openPrice * vol)) / (g_sellBasket.totalLots + vol);
         g_sellBasket.totalLots += vol;
         g_sellBasket.floatingProfit += PositionGetDouble(POSITION_PROFIT);
         g_sellBasket.numberOfOrders++;
         g_sellBasket.isActive = true;
         if(g_sellBasket.basketId == 0)
            g_sellBasket.basketId = BuildBasketId(g_sellBasket.direction);
         if(g_sellBasket.creationTime == 0)
            g_sellBasket.creationTime = (datetime)PositionGetInteger(POSITION_TIME);

         if(g_sellBasket.lastAddPrice == 0.0 || openPrice < g_sellBasket.lastAddPrice)
            g_sellBasket.lastAddPrice = openPrice;
      }
   }

   if(g_buyBasket.isActive)
   {
      g_buyBasket.peakProfit = MathMax(g_buyBasket.peakProfit, g_buyBasket.floatingProfit);
      if(g_buyBasket.floatingProfit >= InpBasketTrailingStart)
         g_buyBasket.trailingArmed = true;
   }

   if(g_sellBasket.isActive)
   {
      g_sellBasket.peakProfit = MathMax(g_sellBasket.peakProfit, g_sellBasket.floatingProfit);
      if(g_sellBasket.floatingProfit >= InpBasketTrailingStart)
         g_sellBasket.trailingArmed = true;
   }
}

bool CloseBasket(const int direction)
{
   bool allClosed = true;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      int type = (int)PositionGetInteger(POSITION_TYPE);
      if(direction > 0 && type != POSITION_TYPE_BUY) continue;
      if(direction < 0 && type != POSITION_TYPE_SELL) continue;

      if(!g_trade.PositionClose(ticket))
         allClosed = false;
   }

   if(allClosed)
      g_daily.totalBasketsClosed++;

   return allClosed;
}

void ManageBasketClose()
{
   if(g_buyBasket.isActive)
   {
      double tpMoney = InpBasketTakeProfitMoney;
      if(g_market.isPanic)
         tpMoney *= 0.6;

      bool closeByTarget = g_buyBasket.floatingProfit >= tpMoney;
      bool closeByTrail = false;
      if(InpBasketTrailingEnabled && g_buyBasket.trailingArmed)
      {
         double trailLevel = g_buyBasket.peakProfit - InpBasketTrailingStep;
         closeByTrail = (g_buyBasket.floatingProfit <= trailLevel);
      }

      bool closeByStagnation = (TimeCurrent() - g_buyBasket.creationTime) > (InpBasketMaxLifetimeMinutes * 60);

      if(closeByTarget || closeByTrail || closeByStagnation)
      {
         if(InpDebugMode)
            Print("Closing basket ", g_buyBasket.basketId, " reason=", (closeByTarget ? "TP" : (closeByTrail ? "TRAIL" : "TIME")));
         CloseBasket(g_buyBasket.direction);
      }
   }

   if(g_sellBasket.isActive)
   {
      double tpMoney = InpBasketTakeProfitMoney;
      if(g_market.isPanic)
         tpMoney *= 0.6;

      bool closeByTarget = g_sellBasket.floatingProfit >= tpMoney;
      bool closeByTrail = false;
      if(InpBasketTrailingEnabled && g_sellBasket.trailingArmed)
      {
         double trailLevel = g_sellBasket.peakProfit - InpBasketTrailingStep;
         closeByTrail = (g_sellBasket.floatingProfit <= trailLevel);
      }

      bool closeByStagnation = (TimeCurrent() - g_sellBasket.creationTime) > (InpBasketMaxLifetimeMinutes * 60);

      if(closeByTarget || closeByTrail || closeByStagnation)
      {
         if(InpDebugMode)
            Print("Closing basket ", g_sellBasket.basketId, " reason=", (closeByTarget ? "TP" : (closeByTrail ? "TRAIL" : "TIME")));
         CloseBasket(g_sellBasket.direction);
      }
   }
}

void ManageGridOrders()
{
   MqlTick tick;
   SymbolInfoTick(_Symbol, tick);

   if(g_buyBasket.isActive && g_buyBasket.numberOfOrders < InpMaxGridOrdersPerSide)
   {
      if(MathAbs(g_buyBasket.floatingProfit) < InpMaxBasketLossMoney)
      {
         double refPrice = (g_buyBasket.lastAddPrice > 0.0 ? g_buyBasket.lastAddPrice : g_buyBasket.weightedAvgPrice);
         if(refPrice > 0.0)
         {
            double adverse = (refPrice - tick.bid) / _Point;
            if(adverse >= InpGridStepPoints)
               OpenGridTrade(g_buyBasket.direction, g_buyBasket.basketId);
         }
      }
   }

   if(g_sellBasket.isActive && g_sellBasket.numberOfOrders < InpMaxGridOrdersPerSide)
   {
      if(MathAbs(g_sellBasket.floatingProfit) < InpMaxBasketLossMoney)
      {
         double refPrice = (g_sellBasket.lastAddPrice > 0.0 ? g_sellBasket.lastAddPrice : g_sellBasket.weightedAvgPrice);
         if(refPrice > 0.0)
         {
            double adverse = (tick.ask - refPrice) / _Point;
            if(adverse >= InpGridStepPoints)
               OpenGridTrade(g_sellBasket.direction, g_sellBasket.basketId);
         }
      }
   }
}

void ManageEmergencyExit()
{
   if(g_buyBasket.isActive && MathAbs(g_buyBasket.floatingProfit) >= InpMaxBasketLossMoney)
   {
      g_disableNewEntries = true;
      CloseBasket(+1);
   }

   if(g_sellBasket.isActive && MathAbs(g_sellBasket.floatingProfit) >= InpMaxBasketLossMoney)
   {
      g_disableNewEntries = true;
      CloseBasket(-1);
   }
}

void UpdateDailyStats()
{
   string nowKey = DateKey(TimeCurrent());
   if(g_daily.todayDate != nowKey)
   {
      g_daily.Reset(nowKey);
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      g_tradingHaltedToday = false;
      g_disableNewEntries = false;
      g_lightMode = false;
   }

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_daily.dailyPnL = equity - g_dayStartEquity;

   double dd = (balance > 0.0 ? (balance - equity) / balance * 100.0 : 0.0);
   g_daily.maxFloatingDD = MathMax(g_daily.maxFloatingDD, dd);

   if(g_daily.closedLots >= InpDailyTargetClosedLots && InpDailyLightModeAfterTarget)
      g_lightMode = true;
}

void ApplyKillSwitch()
{
   if(!CheckDailyLossCap())
      g_tradingHaltedToday = true;

   if(g_daily.maxFloatingDD >= InpMaxFloatingDDPercent)
      g_disableNewEntries = true;

   if(CountOpenOrders() >= InpMaxOpenOrders)
      g_disableNewEntries = true;

   if(g_market.isPanic && InpEnablePanicModeStop)
      g_disableNewEntries = true;
}

void DrawDebugInfoOnChart()
{
   if(!InpDebugMode)
      return;

   string txt = StringFormat(
      "LotFarm EA | Spread: %.1f | ATR: %.1f | Slope: %.2f\n"
      "Panic: %s | Trending: %s | RangeOK: %s\n"
      "OpenedLots: %.2f | ClosedLots: %.2f | Trades: %d | BasketsClosed: %d\n"
      "FloatingDDMax: %.2f%% | DailyPnL: %.2f | OpenOrders: %d\n"
      "BuyBasket: %d / %.2f | SellBasket: %d / %.2f | LightMode: %s",
      g_market.spread,
      g_market.atr,
      g_market.trendSlope,
      (g_market.isPanic ? "YES" : "NO"),
      (g_market.isTrending ? "YES" : "NO"),
      (g_market.isRangeFriendly ? "YES" : "NO"),
      g_daily.openedLots,
      g_daily.closedLots,
      g_daily.totalTrades,
      g_daily.totalBasketsClosed,
      g_daily.maxFloatingDD,
      g_daily.dailyPnL,
      CountOpenOrders(),
      g_buyBasket.numberOfOrders,
      g_buyBasket.floatingProfit,
      g_sellBasket.numberOfOrders,
      g_sellBasket.floatingProfit,
      (g_lightMode ? "YES" : "NO")
   );

   Comment(txt);
}

void TryEntries()
{
   if(!CanOpenNewTrade())
      return;

   bool buySignal = false;
   bool sellSignal = false;

   if(InpEntryMode == ENTRY_MEAN_REVERSION || InpEntryMode == ENTRY_HYBRID)
   {
      buySignal = buySignal || CheckMeanReversionBuySignal();
      sellSignal = sellSignal || CheckMeanReversionSellSignal();
   }

   if(InpEntryMode == ENTRY_BREAKOUT || InpEntryMode == ENTRY_HYBRID)
   {
      buySignal = buySignal || CheckMicroBreakoutBuySignal();
      sellSignal = sellSignal || CheckMicroBreakoutSellSignal();
   }

   if(InpEnableBuy && buySignal && !g_buyBasket.isActive)
      OpenInitialTrade(+1);

   if(InpEnableSell && sellSignal && !g_sellBasket.isActive)
      OpenInitialTrade(-1);
}

void ProcessTick()
{
   UpdateDailyStats();
   UpdateMarketState();
   UpdateBasketStats();

   ApplyKillSwitch();
   ManageEmergencyExit();
   ManageBasketClose();
   ManageGridOrders();

   if(IsNewBar())
      TryEntries();

   DrawDebugInfoOnChart();
}

bool InitIndicators()
{
   g_rsiHandle = iRSI(_Symbol, _Period, InpRSI_Period, PRICE_CLOSE);
   g_emaHandle = iMA(_Symbol, _Period, InpEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   g_bbHandle  = iBands(_Symbol, _Period, InpBollinger_Period, 0, InpBollinger_Deviation, PRICE_CLOSE);
   g_atrHandle = iATR(_Symbol, _Period, InpATR_Period);

   return (g_rsiHandle != INVALID_HANDLE && g_emaHandle != INVALID_HANDLE && g_bbHandle != INVALID_HANDLE && g_atrHandle != INVALID_HANDLE);
}

void ReleaseIndicators()
{
   if(g_rsiHandle != INVALID_HANDLE) IndicatorRelease(g_rsiHandle);
   if(g_emaHandle != INVALID_HANDLE) IndicatorRelease(g_emaHandle);
   if(g_bbHandle  != INVALID_HANDLE) IndicatorRelease(g_bbHandle);
   if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
}

void OnDealUpdate(const MqlTradeTransaction &trans)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   if(!HistoryDealSelect(trans.deal))
      return;

   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol)
      return;
   if((long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagicNumber)
      return;

   datetime dealTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
   if(DateKey(dealTime) != g_daily.todayDate)
      return;

   long entryType = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   double volume = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT) + HistoryDealGetDouble(trans.deal, DEAL_SWAP) + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);

   if(entryType == DEAL_ENTRY_OUT || entryType == DEAL_ENTRY_OUT_BY)
   {
      g_daily.closedLots += volume;
      g_daily.dailyPnL += profit;
   }
}

// =========================================================
// EA LIFECYCLE
// =========================================================
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

   Print("XAUUSD_HybridLotFarm_EA initialized (single-file mode).");
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

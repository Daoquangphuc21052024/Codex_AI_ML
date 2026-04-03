#pragma once

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

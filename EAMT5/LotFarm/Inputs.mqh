#pragma once

input group "[A] General"
input long   InpMagicNumber          = 260401;
input bool   InpEnableBuy            = true;
input bool   InpEnableSell           = true;
input double InpMaxSpread            = 80.0; // points
input bool   InpUseTradingHours      = true;
input int    InpStartHour            = 7;
input int    InpEndHour              = 22;
input int    InpSlippagePoints       = 30;
input bool   InpDebugMode            = true;

input group "[B] Entry Settings"
input EntryMode InpEntryMode         = ENTRY_HYBRID;
input int    InpRSI_Period           = 8;
input double InpRSI_OB               = 72.0;
input double InpRSI_OS               = 28.0;
input int    InpEMA_Period           = 55;
input int    InpBollinger_Period     = 20;
input double InpBollinger_Deviation  = 2.0;
input int    InpBreakoutLookback     = 12;

input group "[C] Grid Settings"
input double   InpBaseLot            = 0.02;
input double   InpGridStepPoints     = 180.0;
input int      InpMaxGridOrdersPerSide = 6;
input GridMode InpGridMode           = GRID_SOFT_MULT;
input double   InpSoftMultiplier     = 1.18;
input double   InpAdditiveLotStep    = 0.01;
input double   InpMaxLotPerOrder     = 0.20;
input double   InpMaxTotalExposureLots = 2.00;

input group "[D] Basket Close Settings"
input double InpBasketTakeProfitMoney   = 8.0;
input bool   InpBasketTrailingEnabled   = true;
input double InpBasketTrailingStart     = 10.0;
input double InpBasketTrailingStep      = 2.0;
input int    InpBasketMaxLifetimeMinutes = 240;

input group "[E] Risk Filters"
input int    InpATR_Period            = 14;
input double InpATR_DangerThreshold   = 7.5;
input double InpTrendSlopeThreshold   = 2.0;
input double InpDistanceDangerThreshold = 10.0;
input double InpPanicCandleMultiplier = 2.2;

input group "[F] Daily Governor"
input double InpDailyTargetClosedLots = 8.0;
input double InpDailyMaxOpenedLots    = 16.0;
input int    InpDailyMaxTrades        = 280;
input bool   InpDailyLightModeAfterTarget = true;

input group "[G] Kill Switch"
input double InpDailyEquityStopPercent = 6.0;
input double InpMaxBasketLossMoney     = 70.0;
input double InpMaxFloatingDDPercent   = 10.0;
input int    InpMaxOpenOrders          = 20;
input bool   InpEnablePanicModeStop    = true;

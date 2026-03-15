//+------------------------------------------------------------------+
//| ICT_SMC_EA_V3_step1.mq5                                          |
//| ICT Smart Money Concept Expert Advisor                           |
//| Refactor v3 - single-file architecture                           |
//+------------------------------------------------------------------+
#property copyright "ICT SMC EA v3.10"
#property link      ""
#property version   "3.10"
#property description "ICT Strategy v3 step1: safer narrative state-machine scaffold"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//============================================================
// ENUMS
//============================================================
enum ENUM_RISK_MODE
{
   RISK_FIXED_LOT = 0,
   RISK_PERCENT   = 1
};

enum ENUM_TRADE_SESSION
{
   SESSION_LONDON  = 0,
   SESSION_NEWYORK = 1,
   SESSION_BOTH    = 2
};

enum ENUM_ENTRY_MODE
{
   ENTRY_MID_FVG        = 0,
   ENTRY_NEAR_FVG       = 1,
   ENTRY_CANDLE_CONFIRM = 2
};

enum ENUM_TRAILING_MODE
{
   TRAIL_ATR   = 0,
   TRAIL_SWING = 1
};

enum ENUM_BIAS
{
   BIAS_NONE    =  0,
   BIAS_BULLISH =  1,
   BIAS_BEARISH = -1
};

enum ENUM_STRUCTURE_TYPE
{
   STRUCT_NONE  = 0,
   STRUCT_BOS   = 1,
   STRUCT_CHOCH = 2
};

enum ENUM_SWING_TIER
{
   SWING_INTERNAL = 0,
   SWING_EXTERNAL = 1
};

enum ENUM_SETUP_STATE
{
   SETUP_IDLE                = 0,
   SETUP_WAIT_SWEEP          = 1,
   SETUP_SWEEP_CONFIRMED     = 2,
   SETUP_SHIFT_CONFIRMED     = 3,
   SETUP_DISPLACEMENT_READY  = 4,
   SETUP_ENTRY_READY         = 5,
   SETUP_ORDER_PLACED        = 6,
   SETUP_INVALID             = 7
};

enum ENUM_ORDER_EXECUTION
{
   EXEC_MARKET = 0,
   EXEC_PENDING_LIMIT = 1
};

enum ENUM_STRUCTURE_EVENT
{
   STRUCT_EVENT_NONE  = 0,
   STRUCT_EVENT_BOS   = 1,
   STRUCT_EVENT_CHOCH = 2,
   STRUCT_EVENT_MSS   = 3
};

enum ENUM_INVALID_REASON
{
   INVALID_NONE                 = 0,
   INVALID_TIMEOUT_SWEEP        = 1,
   INVALID_TIMEOUT_SHIFT        = 2,
   INVALID_TIMEOUT_RETRACE      = 3,
   INVALID_SWEEP_FAILED         = 4,
   INVALID_SHIFT_FAILED         = 5,
   INVALID_DISPLACEMENT_FAILED  = 6,
   INVALID_RETRACE_TOO_DEEP     = 7,
   INVALID_SESSION_FILTER       = 8,
   INVALID_BIAS_FILTER          = 9
};

//============================================================
// STRUCTS
//============================================================
struct SwingPoint
{
   datetime time;
   double   price;
   bool     isHigh;
   bool     broken;
   int      barIndex;
   int      tier;

   // Reserved for future use
   double   strength;
   bool     isEqualGroup;
   int      liquidityTag;
   int      absoluteIndex;
};

struct MarketStructure
{
   ENUM_STRUCTURE_TYPE type;
   datetime            time;
   double              level;
   int                 direction;
   bool                valid;
   int                 breakBar;
   int                 sourceTier;
};

struct LiquiditySweep
{
   bool     valid;
   datetime time;
   double   liquidityLevel;
   double   sweepHigh;
   double   sweepLow;
   int      direction;
   int      barIndex;
   int      tier;
   bool     rejectionStrong;
};

struct LiquidityPool
{
   bool     valid;
   bool     isHighSide;
   double   topPrice;
   double   bottomPrice;
   datetime leftTime;
   datetime rightTime;
   int      swingCount;
   bool     swept;
   datetime sweepTime;
   double   sweepPrice;
   bool     active;
};

struct FVGZone
{
   bool     valid;
   bool     active;
   bool     mitigated;
   datetime timeStart;
   datetime timeEnd;
   double   upper;
   double   lower;
   int      direction;
   int      barIndex;

   // Reserved for future use
   int      linkedLegId;
   bool     fresh;
   double   mitigatedPercent;
   int      entryQuality;
};

struct OTEZone
{
   bool   valid;
   double upper;
   double lower;
   int    direction;
};

struct DisplacementInfo
{
   bool     valid;
   datetime time;
   int      direction;
   double   bodySize;
   double   atrValue;
   int      barIndex;
   double   impulseStart;
   double   impulseEnd;
   int      impulseStartBar;
   int      impulseEndBar;

   // Reserved for future use
   datetime legStartTime;
   datetime legEndTime;
   double   totalRangePoints;
   double   bodyDominance;
   bool     hasImpulse;
   bool     hasFVG;
   int      linkedStructureEvent;
};

struct TradeSetup
{
   bool                 valid;
   int                  direction;
   datetime             signalTime;
   double               entryPrice;
   double               stopLoss;
   double               takeProfit1;
   double               takeProfit2;
   double               sweepLevel;
   double               fvgHigh;
   double               fvgLow;
   double               oteHigh;
   double               oteLow;
   string               reason;
   double               lotSize;
   ENUM_ORDER_EXECUTION execType;
   datetime             expiryTime;
};

struct SetupContext
{
   ENUM_SETUP_STATE state;
   int              direction;
   datetime         startedAt;
   int              startBarShift;
   int              barsInState;

   bool             active;
   datetime         createdTime;
   datetime         lastUpdateTime;
   ENUM_INVALID_REASON invalidReasonCode;
   string           invalidReasonText;

   LiquidityPool    pool;
   LiquiditySweep   sweep;
   MarketStructure  structure;
   DisplacementInfo displacement;
   FVGZone          fvg;
   OTEZone          ote;
   TradeSetup       trade;

   bool             usePending;
   bool             entryTouched;
   bool             invalid;

   int              barsSinceSweep;
   int              barsSinceShift;
   int              barsSinceDisplacement;
   int              barsSinceEntryReady;

   ulong            orderTicket;
   ulong            positionTicket;
};

struct PositionRuntime
{
   ulong  ticket;
   double initialVolume;
   bool   partialDone;

   // Reserved for future use
   int    direction;
   bool   partialDone1;
   bool   partialDone2;
   bool   movedToBE;
   bool   trailingActive;
   double initialRiskPoints;
   datetime openTime;
   bool   active;
};

//============================================================
// INPUT PARAMETERS
//============================================================
input group "=== GENERAL ==="
input long   InpMagicNumber       = 202501;
input string InpComment           = "ICT_SMC";
input bool   InpAllowBuy          = true;
input bool   InpAllowSell         = true;
input bool   InpOneTradePerSymbol = true;
input bool   InpDebugMode         = true;

input group "=== TIMEFRAMES ==="
input ENUM_TIMEFRAMES InpSignalTF = PERIOD_M15;
input ENUM_TIMEFRAMES InpBiasTF   = PERIOD_H1;

input group "=== SWING & STRUCTURE ==="
input int InpSwingLeft            = 3;
input int InpSwingRight           = 3;
input int InpEqualTolerancePoints = 5;
input int InpMaxSwingsToKeep      = 40;
input int InpMSSLookbackBars      = 8;
input int InpSetupExpiryBars      = 10;
input int InpSweepExpiryBars      = 4;
input int InpDisplacementMaxBars  = 3;

input group "=== DISPLACEMENT ==="
input int    InpATRPeriod            = 14;
input double InpDisplacementATRMult  = 1.3;
input double InpMinBodyRangeRatio    = 0.60;
input double InpImpulseMinATR        = 1.8;

input group "=== FVG & OTE ==="
input bool             InpUseFVG            = true;
input bool             InpUseOTE            = false;
input double           InpOTELevel1         = 0.62;
input double           InpOTELevel2         = 0.79;
input ENUM_ENTRY_MODE  InpEntryMode         = ENTRY_MID_FVG;
input int              InpMinFVGSizePoints  = 5;
input bool             InpAllowPending      = true;
input int              InpPendingExpiryBars = 6;

input group "=== SESSION ==="
input bool               InpUseSessionFilter = true;
input ENUM_TRADE_SESSION InpTradeSession     = SESSION_BOTH;
input int                InpLondonStart      = 8;
input int                InpLondonEnd        = 12;
input int                InpNewYorkStart     = 13;
input int                InpNewYorkEnd       = 18;

input group "=== RISK MANAGEMENT ==="
input ENUM_RISK_MODE     InpRiskMode         = RISK_PERCENT;
input double             InpFixedLot         = 0.01;
input double             InpRiskPercent      = 1.0;
input int                InpSLBufferPoints   = 10;
input double             InpRR_TP1           = 2.0;
input double             InpRR_TP2           = 4.0;
input bool               InpUsePartialClose  = true;
input double             InpPartialPercent   = 50.0;
input bool               InpMoveToBEAfterTP1 = true;
input int                InpBESafetyPoints   = 2;
input bool               InpUseTrailing      = false;
input ENUM_TRAILING_MODE InpTrailingMode     = TRAIL_ATR;
input double             InpTrailingATRMult  = 1.5;

input group "=== FILTERS ==="
input int  InpMaxSpreadPoints          = 30;
input bool InpUsePremiumDiscountFilter = false;

input group "=== VISUALIZATION ==="
input bool InpShowSwings   = true;
input bool InpShowBOSCHoCH = true;
input bool InpShowSweep    = true;
input bool InpShowFVG      = true;
input bool InpShowOTE      = true;
input bool InpShowEntry    = true;

//============================================================
// GLOBAL VARIABLES
//============================================================
CTrade        g_trade;
CPositionInfo g_position;
CSymbolInfo   g_symbol;

SwingPoint g_internalSwings[];
int        g_internalCount = 0;
SwingPoint g_externalSwings[];
int        g_externalCount = 0;
SwingPoint g_htfSwings[];
int        g_htfSwingCount = 0;

MarketStructure  g_lastStructure;
LiquiditySweep   g_lastSweep;
FVGZone          g_lastFVG;
DisplacementInfo g_lastDisplacement;
OTEZone          g_oteZone;
TradeSetup       g_currentSetup;

SetupContext     g_buyCtx;
SetupContext     g_sellCtx;
PositionRuntime  g_positionRuntime;

datetime g_lastBarTime    = 0;
datetime g_lastHTFBarTime = 0;
ENUM_BIAS g_htfBias = BIAS_NONE;

int    g_atrHandle    = INVALID_HANDLE;
int    g_htfAtrHandle = INVALID_HANDLE;
string g_objPrefix;
string g_logPrefix = "[ICT_V3]";

//============================================================
// HELPERS (safety / formatting)
//============================================================
string SetupStateToString(ENUM_SETUP_STATE state)
{
   switch(state)
   {
      case SETUP_IDLE: return "IDLE";
      case SETUP_WAIT_SWEEP: return "WAIT_SWEEP";
      case SETUP_SWEEP_CONFIRMED: return "SWEEP_CONFIRMED";
      case SETUP_SHIFT_CONFIRMED: return "SHIFT_CONFIRMED";
      case SETUP_DISPLACEMENT_READY: return "DISPLACEMENT_READY";
      case SETUP_ENTRY_READY: return "ENTRY_READY";
      case SETUP_ORDER_PLACED: return "ORDER_PLACED";
      case SETUP_INVALID: return "INVALID";
   }
   return "UNKNOWN";
}

string StructureTypeToString(ENUM_STRUCTURE_TYPE type)
{
   switch(type)
   {
      case STRUCT_BOS: return "BOS";
      case STRUCT_CHOCH: return "CHOCH";
      default: return "NONE";
   }
}

string BiasToString(ENUM_BIAS bias)
{
   switch(bias)
   {
      case BIAS_BULLISH: return "BULLISH";
      case BIAS_BEARISH: return "BEARISH";
      default: return "NONE";
   }
}

string InvalidReasonToString(ENUM_INVALID_REASON reason)
{
   switch(reason)
   {
      case INVALID_TIMEOUT_SWEEP: return "TIMEOUT_SWEEP";
      case INVALID_TIMEOUT_SHIFT: return "TIMEOUT_SHIFT";
      case INVALID_TIMEOUT_RETRACE: return "TIMEOUT_RETRACE";
      case INVALID_SWEEP_FAILED: return "SWEEP_FAILED";
      case INVALID_SHIFT_FAILED: return "SHIFT_FAILED";
      case INVALID_DISPLACEMENT_FAILED: return "DISPLACEMENT_FAILED";
      case INVALID_RETRACE_TOO_DEEP: return "RETRACE_TOO_DEEP";
      case INVALID_SESSION_FILTER: return "SESSION_FILTER";
      case INVALID_BIAS_FILTER: return "BIAS_FILTER";
      default: return "NONE";
   }
}

double NormalizePrice(double price) { return NormalizeDouble(price, _Digits); }

bool HasEnoughBars(ENUM_TIMEFRAMES tf, int needed)
{
   if(needed < 1) needed = 1;
   return (Bars(_Symbol, tf) >= needed);
}

bool IsValidShift(ENUM_TIMEFRAMES tf, int shift)
{
   if(shift < 0) return false;
   return (shift < Bars(_Symbol, tf));
}

bool ReadCandle(ENUM_TIMEFRAMES tf, int shift, datetime &t, double &o, double &h, double &l, double &c)
{
   if(!IsValidShift(tf, shift)) return false;
   t = iTime(_Symbol, tf, shift);
   if(t <= 0) return false;
   o = iOpen(_Symbol, tf, shift);
   h = iHigh(_Symbol, tf, shift);
   l = iLow(_Symbol, tf, shift);
   c = iClose(_Symbol, tf, shift);
   return true;
}

string DirectionToText(int direction) { return (direction == 1 ? "BUY" : "SELL"); }
void WriteDebugLog(string message)
{
   if(!InpDebugMode) return;
   Print(g_logPrefix, "[", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "] ", message);
}

//============================================================
// INITIALIZATION
//============================================================
int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(10);
   ConfigureTradeFilling();

   g_symbol.Name(_Symbol);
   g_symbol.RefreshRates();

   g_objPrefix = "ICTV3_" + IntegerToString(InpMagicNumber) + "_";

   g_atrHandle = iATR(_Symbol, InpSignalTF, InpATRPeriod);
   if(g_atrHandle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create ATR handle for Signal TF");
      return INIT_FAILED;
   }

   g_htfAtrHandle = iATR(_Symbol, InpBiasTF, InpATRPeriod);
   if(g_htfAtrHandle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create ATR handle for Bias TF");
      return INIT_FAILED;
   }

   ArrayResize(g_internalSwings, InpMaxSwingsToKeep);
   ArrayResize(g_externalSwings, InpMaxSwingsToKeep);
   ArrayResize(g_htfSwings,      InpMaxSwingsToKeep);

   ResetAllStructures();
   Print("=== ICT SMC EA v3 initialized ===");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_atrHandle    != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
   if(g_htfAtrHandle != INVALID_HANDLE) IndicatorRelease(g_htfAtrHandle);
   DeleteAllICTObjects();
   Print("ICT SMC EA v3 deinitialized. Reason: ", reason);
}

void InitSetupContext(SetupContext &ctx, int direction)
{
   ZeroMemory(ctx);
   ctx.state             = SETUP_WAIT_SWEEP;
   ctx.direction         = direction;
   ctx.active            = true;
   ctx.createdTime       = TimeCurrent();
   ctx.invalidReasonCode = INVALID_NONE;
   ctx.invalidReasonText = "";
}

void ResetAllStructures()
{
   ZeroMemory(g_lastStructure);
   ZeroMemory(g_lastSweep);
   ZeroMemory(g_lastFVG);
   ZeroMemory(g_lastDisplacement);
   ZeroMemory(g_oteZone);
   ZeroMemory(g_currentSetup);
   ZeroMemory(g_buyCtx);
   ZeroMemory(g_sellCtx);
   ZeroMemory(g_positionRuntime);

   g_internalCount = 0;
   g_externalCount = 0;
   g_htfSwingCount = 0;
   g_htfBias       = BIAS_NONE;

   InitSetupContext(g_buyCtx,  1);
   InitSetupContext(g_sellCtx, -1);
}

void MoveSetupState(SetupContext &ctx, ENUM_SETUP_STATE newState)
{
   if(ctx.state == newState) return;
   WriteDebugLog("Ctx " + DirectionToText(ctx.direction) + " state " + SetupStateToString(ctx.state) + " -> " + SetupStateToString(newState));
   ctx.state          = newState;
   ctx.barsInState    = 0;
   ctx.lastUpdateTime = TimeCurrent();
}

void InvalidateSetupContext(SetupContext &ctx, ENUM_INVALID_REASON reason, string reasonText)
{
   ctx.invalid           = true;
   ctx.active            = false;
   ctx.invalidReasonCode = reason;
   ctx.invalidReasonText = reasonText;
   MoveSetupState(ctx, SETUP_INVALID);
   WriteDebugLog("Ctx " + DirectionToText(ctx.direction) + " invalid: " + InvalidReasonToString(reason) + " / " + reasonText);
}

void ResetContextToWaitSweep(SetupContext &ctx)
{
   int direction = ctx.direction;
   ENUM_INVALID_REASON lastReason = ctx.invalidReasonCode;
   string lastReasonText = ctx.invalidReasonText;
   InitSetupContext(ctx, direction);
   ctx.invalidReasonCode = lastReason;
   ctx.invalidReasonText = lastReasonText;
}

//============================================================
// MAIN
//============================================================
void OnTick()
{
   g_symbol.RefreshRates();

   datetime currentBarTime = iTime(_Symbol, InpSignalTF, 0);
   datetime currentHTFBarTime = iTime(_Symbol, InpBiasTF, 0);
   bool isNewSignalBar = (currentBarTime != g_lastBarTime);
   bool isNewHTFBar = (currentHTFBarTime != g_lastHTFBarTime);

   ManageOrdersAndPositions();

   if(isNewHTFBar)
   {
      g_lastHTFBarTime = currentHTFBarTime;
      DetermineHTFBias();
   }

   if(!isNewSignalBar)
      return;

   g_lastBarTime = currentBarTime;

   int barsNeeded = MathMax(MathMax(InpSwingLeft + InpSwingRight + 10, InpATRPeriod + 10), 50);
   if(!HasEnoughBars(InpSignalTF, barsNeeded))
   {
      WriteDebugLog("Insufficient bars on Signal TF");
      return;
   }

   DetectSwingPoints();
   DetectMarketStructure();
   DetectLiquiditySweep();
   DetectDisplacement();
   if(InpUseFVG) DetectFVG();
   if(InpUseOTE) CalculateOTEZone();

   UpdateSetupContext(g_buyCtx);
   UpdateSetupContext(g_sellCtx);

   BuildTradeSetup();
   if(g_currentSetup.valid && ValidateTradeSetup())
   {
      g_currentSetup.lotSize = CalculateLotSize(g_currentSetup.entryPrice, g_currentSetup.stopLoss);
      if(g_currentSetup.lotSize > 0.0)
         ExecuteTradeSetup(g_currentSetup);
   }

   DrawICTObjects();
}

//============================================================
// SWING/STRUCTURE/SWEEP/DISPLACEMENT/FVG/OTE/BIAS
//============================================================
// NOTE: Logic intentionally preserved. Only guard/refactor style changed.

void DetectSwingPoints();
void DetectTierSwing(int tier, int leftBars, int rightBars);
bool SwingExists(SwingPoint &arr[], int count, datetime t, bool isHigh, int tier);
void AddSwingToArray(SwingPoint &arr[], int &count, datetime t, double price, bool isHigh, int barIdx, int tier);
bool GetLastUnbrokenSwing(int tier, bool wantHigh, SwingPoint &out);
bool GetLastTwoSwings(int tier, bool wantHigh, SwingPoint &latest, SwingPoint &previous);
void MarkSwingBroken(int tier, datetime t, bool isHigh);
int  DetermineInternalTrend();
void DetectMarketStructure();
void DetectLiquiditySweep();
double GetATRValue(int shift);
void DetectDisplacement();
void BuildImpulseLeg(DisplacementInfo &disp);
void DetectFVG();
void CalculateOTEZone();
void DetermineHTFBias();
bool GetLastTwoSwingFromArray(SwingPoint &arr[], int count, bool wantHigh, SwingPoint &latest, SwingPoint &previous);

// -- implementations copied/guarded
void DetectSwingPoints(){ DetectTierSwing(SWING_INTERNAL, MathMax(1, InpSwingLeft / 2), MathMax(1, InpSwingRight / 2)); DetectTierSwing(SWING_EXTERNAL, InpSwingLeft, InpSwingRight); }

void DetectTierSwing(int tier, int leftBars, int rightBars)
{
   int pivotBar = MathMax(1, rightBars);
   if(!IsValidShift(InpSignalTF, pivotBar + leftBars) || !IsValidShift(InpSignalTF, pivotBar - rightBars)) return;

   double pivotHigh = iHigh(_Symbol, InpSignalTF, pivotBar);
   double pivotLow  = iLow(_Symbol, InpSignalTF, pivotBar);
   datetime pivotTime = iTime(_Symbol, InpSignalTF, pivotBar);

   bool isSwingHigh = true, isSwingLow = true;
   for(int i=1;i<=leftBars;i++){ if(iHigh(_Symbol, InpSignalTF, pivotBar + i) >= pivotHigh) isSwingHigh=false; if(iLow(_Symbol, InpSignalTF, pivotBar + i) <= pivotLow) isSwingLow=false; }
   for(int i=1;i<=rightBars;i++){ if(iHigh(_Symbol, InpSignalTF, pivotBar - i) >= pivotHigh) isSwingHigh=false; if(iLow(_Symbol, InpSignalTF, pivotBar - i) <= pivotLow) isSwingLow=false; }

   if(tier == SWING_INTERNAL)
   {
      if(isSwingHigh && !SwingExists(g_internalSwings,g_internalCount,pivotTime,true,tier))
         AddSwingToArray(g_internalSwings,g_internalCount,pivotTime,pivotHigh,true,pivotBar,tier);
      if(isSwingLow && !SwingExists(g_internalSwings,g_internalCount,pivotTime,false,tier))
         AddSwingToArray(g_internalSwings,g_internalCount,pivotTime,pivotLow,false,pivotBar,tier);
   }
   else
   {
      if(isSwingHigh && !SwingExists(g_externalSwings,g_externalCount,pivotTime,true,tier))
         AddSwingToArray(g_externalSwings,g_externalCount,pivotTime,pivotHigh,true,pivotBar,tier);
      if(isSwingLow && !SwingExists(g_externalSwings,g_externalCount,pivotTime,false,tier))
         AddSwingToArray(g_externalSwings,g_externalCount,pivotTime,pivotLow,false,pivotBar,tier);
   }
}

bool SwingExists(SwingPoint &arr[], int count, datetime t, bool isHigh, int tier){ for(int i=0;i<count;i++) if(arr[i].time==t && arr[i].isHigh==isHigh && arr[i].tier==tier) return true; return false; }
void AddSwingToArray(SwingPoint &arr[], int &count, datetime t, double price, bool isHigh, int barIdx, int tier){ if(count>=InpMaxSwingsToKeep){ for(int i=0;i<count-1;i++) arr[i]=arr[i+1]; count--; } arr[count].time=t; arr[count].price=price; arr[count].isHigh=isHigh; arr[count].broken=false; arr[count].barIndex=barIdx; arr[count].tier=tier; count++; }

bool GetLastUnbrokenSwing(int tier, bool wantHigh, SwingPoint &out)
{
   if(tier == SWING_INTERNAL)
   {
      for(int i=g_internalCount-1;i>=0;i--)
         if(g_internalSwings[i].isHigh==wantHigh && !g_internalSwings[i].broken){ out=g_internalSwings[i]; return true; }
   }
   else
   {
      for(int i=g_externalCount-1;i>=0;i--)
         if(g_externalSwings[i].isHigh==wantHigh && !g_externalSwings[i].broken){ out=g_externalSwings[i]; return true; }
   }
   return false;
}

bool GetLastTwoSwings(int tier, bool wantHigh, SwingPoint &latest, SwingPoint &previous)
{
   int found=0;
   if(tier == SWING_INTERNAL)
   {
      for(int i=g_internalCount-1;i>=0;i--)
      {
         if(g_internalSwings[i].isHigh!=wantHigh) continue;
         if(found==0) latest=g_internalSwings[i];
         else { previous=g_internalSwings[i]; return true; }
         found++;
      }
   }
   else
   {
      for(int i=g_externalCount-1;i>=0;i--)
      {
         if(g_externalSwings[i].isHigh!=wantHigh) continue;
         if(found==0) latest=g_externalSwings[i];
         else { previous=g_externalSwings[i]; return true; }
         found++;
      }
   }
   return false;
}

void MarkSwingBroken(int tier, datetime t, bool isHigh)
{
   if(tier == SWING_INTERNAL)
   {
      for(int i=0;i<g_internalCount;i++)
         if(g_internalSwings[i].time==t && g_internalSwings[i].isHigh==isHigh){ g_internalSwings[i].broken=true; break; }
   }
   else
   {
      for(int i=0;i<g_externalCount;i++)
         if(g_externalSwings[i].time==t && g_externalSwings[i].isHigh==isHigh){ g_externalSwings[i].broken=true; break; }
   }
}

int DetermineInternalTrend()
{
   SwingPoint lastHigh, prevHigh, lastLow, prevLow;
   if(!GetLastTwoSwings(SWING_INTERNAL,true,lastHigh,prevHigh) || !GetLastTwoSwings(SWING_INTERNAL,false,lastLow,prevLow)) return 0;
   bool bullish = (lastHigh.price > prevHigh.price && lastLow.price > prevLow.price);
   bool bearish = (lastHigh.price < prevHigh.price && lastLow.price < prevLow.price);
   if(bullish) return 1; if(bearish) return -1; return 0;
}

void DetectMarketStructure()
{
   g_lastStructure.valid = false;
   if(!IsValidShift(InpSignalTF,1)) return;
   double close1=iClose(_Symbol,InpSignalTF,1); datetime time1=iTime(_Symbol,InpSignalTF,1);
   int internalTrend=DetermineInternalTrend();
   SwingPoint internalHigh,internalLow;
   bool hasInternalHigh=GetLastUnbrokenSwing(SWING_INTERNAL,true,internalHigh);
   bool hasInternalLow=GetLastUnbrokenSwing(SWING_INTERNAL,false,internalLow);

   if(hasInternalHigh && close1 > internalHigh.price)
   {
      g_lastStructure.valid=true; g_lastStructure.time=time1; g_lastStructure.level=internalHigh.price; g_lastStructure.direction=1; g_lastStructure.breakBar=1; g_lastStructure.sourceTier=SWING_INTERNAL;
      g_lastStructure.type=(internalTrend==-1?STRUCT_CHOCH:STRUCT_BOS); MarkSwingBroken(SWING_INTERNAL,internalHigh.time,true); return;
   }
   if(hasInternalLow && close1 < internalLow.price)
   {
      g_lastStructure.valid=true; g_lastStructure.time=time1; g_lastStructure.level=internalLow.price; g_lastStructure.direction=-1; g_lastStructure.breakBar=1; g_lastStructure.sourceTier=SWING_INTERNAL;
      g_lastStructure.type=(internalTrend==1?STRUCT_CHOCH:STRUCT_BOS); MarkSwingBroken(SWING_INTERNAL,internalLow.time,false); return;
   }
}

void DetectLiquiditySweep()
{
   g_lastSweep.valid=false;
   if(!IsValidShift(InpSignalTF,1)) return;
   double high1=iHigh(_Symbol,InpSignalTF,1), low1=iLow(_Symbol,InpSignalTF,1), open1=iOpen(_Symbol,InpSignalTF,1), close1=iClose(_Symbol,InpSignalTF,1);
   double range1=high1-low1, body1=MathAbs(close1-open1); datetime time1=iTime(_Symbol,InpSignalTF,1); double tol=InpEqualTolerancePoints*_Point;
   SwingPoint extLow,extHigh; bool hasExtLow=GetLastUnbrokenSwing(SWING_EXTERNAL,false,extLow), hasExtHigh=GetLastUnbrokenSwing(SWING_EXTERNAL,true,extHigh);

   // Future extension point: aggregate equal highs/lows into LiquidityPool here.
   if(hasExtLow)
   {
      bool wickBelow=(low1<extLow.price-tol), closeBackIn=(close1>extLow.price), rejection=(range1>0 && body1/range1>=0.40 && close1>(low1+range1*0.45));
      if(wickBelow && closeBackIn){ g_lastSweep.valid=true; g_lastSweep.time=time1; g_lastSweep.liquidityLevel=extLow.price; g_lastSweep.sweepLow=low1; g_lastSweep.sweepHigh=high1; g_lastSweep.direction=1; g_lastSweep.barIndex=1; g_lastSweep.tier=SWING_EXTERNAL; g_lastSweep.rejectionStrong=rejection; return; }
   }
   if(hasExtHigh)
   {
      bool wickAbove=(high1>extHigh.price+tol), closeBackIn=(close1<extHigh.price), rejection=(range1>0 && body1/range1>=0.40 && close1<(high1-range1*0.45));
      if(wickAbove && closeBackIn){ g_lastSweep.valid=true; g_lastSweep.time=time1; g_lastSweep.liquidityLevel=extHigh.price; g_lastSweep.sweepLow=low1; g_lastSweep.sweepHigh=high1; g_lastSweep.direction=-1; g_lastSweep.barIndex=1; g_lastSweep.tier=SWING_EXTERNAL; g_lastSweep.rejectionStrong=rejection; return; }
   }
}

double GetATRValue(int shift){ double atrBuf[]; ArraySetAsSeries(atrBuf,true); if(CopyBuffer(g_atrHandle,0,shift,1,atrBuf)<1) return 0.0; return atrBuf[0]; }

void DetectDisplacement()
{
   g_lastDisplacement.valid=false;
   for(int shift=1; shift<=InpDisplacementMaxBars; shift++)
   {
      if(!IsValidShift(InpSignalTF,shift)) continue;
      double atr=GetATRValue(shift); if(atr<=0) continue;
      double o=iOpen(_Symbol,InpSignalTF,shift), c=iClose(_Symbol,InpSignalTF,shift), h=iHigh(_Symbol,InpSignalTF,shift), l=iLow(_Symbol,InpSignalTF,shift);
      double body=MathAbs(c-o), range=h-l, ratio=(range>0?body/range:0); int direction=(c>o?1:-1);
      bool sizeOK=(body>=atr*InpDisplacementATRMult), ratioOK=(ratio>=InpMinBodyRangeRatio), closeNearExtreme=(direction==1?c>=h-range*0.25:c<=l+range*0.25);
      if(sizeOK && ratioOK && closeNearExtreme)
      {
         g_lastDisplacement.valid=true; g_lastDisplacement.time=iTime(_Symbol,InpSignalTF,shift); g_lastDisplacement.direction=direction; g_lastDisplacement.bodySize=body; g_lastDisplacement.atrValue=atr; g_lastDisplacement.barIndex=shift;
         BuildImpulseLeg(g_lastDisplacement); return;
      }
   }
}

void BuildImpulseLeg(DisplacementInfo &disp)
{
   disp.impulseStart=0; disp.impulseEnd=0; disp.impulseStartBar=-1; disp.impulseEndBar=disp.barIndex;
   int barsCount=Bars(_Symbol,InpSignalTF); if(barsCount<5) return;
   int startShift=MathMin(disp.barIndex + InpMSSLookbackBars, barsCount - 2);

   if(disp.direction==1)
   {
      double minLow=DBL_MAX, maxHigh=-DBL_MAX; int minBar=-1, maxBar=disp.barIndex;
      for(int i=startShift;i>=disp.barIndex;i--){ double lv=iLow(_Symbol,InpSignalTF,i), hv=iHigh(_Symbol,InpSignalTF,i); if(lv<minLow){minLow=lv; minBar=i;} if(hv>maxHigh){maxHigh=hv; maxBar=i;} }
      disp.impulseStart=minLow; disp.impulseEnd=maxHigh; disp.impulseStartBar=minBar; disp.impulseEndBar=maxBar;
   }
   else
   {
      double maxHigh=-DBL_MAX, minLow=DBL_MAX; int maxBar=-1, minBar=disp.barIndex;
      for(int i=startShift;i>=disp.barIndex;i--){ double hv=iHigh(_Symbol,InpSignalTF,i), lv=iLow(_Symbol,InpSignalTF,i); if(hv>maxHigh){maxHigh=hv; maxBar=i;} if(lv<minLow){minLow=lv; minBar=i;} }
      disp.impulseStart=maxHigh; disp.impulseEnd=minLow; disp.impulseStartBar=maxBar; disp.impulseEndBar=minBar;
   }
}

void DetectFVG()
{
   g_lastFVG.valid=false; if(!g_lastDisplacement.valid) return;
   int center=g_lastDisplacement.barIndex; if(center+1>=Bars(_Symbol,InpSignalTF) || center-1<0) return;
   int left=center+1, right=center-1; double minSize=InpMinFVGSizePoints*_Point;
   double leftHigh=iHigh(_Symbol,InpSignalTF,left), leftLow=iLow(_Symbol,InpSignalTF,left), rightHigh=iHigh(_Symbol,InpSignalTF,right), rightLow=iLow(_Symbol,InpSignalTF,right);
   if(g_lastDisplacement.direction==1)
   {
      if(rightLow>leftHigh && (rightLow-leftHigh)>=minSize){ g_lastFVG.valid=true; g_lastFVG.active=true; g_lastFVG.mitigated=false; g_lastFVG.fresh=true; g_lastFVG.mitigatedPercent=0; g_lastFVG.timeStart=iTime(_Symbol,InpSignalTF,left); g_lastFVG.timeEnd=iTime(_Symbol,InpSignalTF,right); g_lastFVG.lower=leftHigh; g_lastFVG.upper=rightLow; g_lastFVG.direction=1; g_lastFVG.barIndex=center; }
   }
   else
   {
      if(rightHigh<leftLow && (leftLow-rightHigh)>=minSize){ g_lastFVG.valid=true; g_lastFVG.active=true; g_lastFVG.mitigated=false; g_lastFVG.fresh=true; g_lastFVG.mitigatedPercent=0; g_lastFVG.timeStart=iTime(_Symbol,InpSignalTF,left); g_lastFVG.timeEnd=iTime(_Symbol,InpSignalTF,right); g_lastFVG.lower=rightHigh; g_lastFVG.upper=leftLow; g_lastFVG.direction=-1; g_lastFVG.barIndex=center; }
   }
}

void CalculateOTEZone()
{
   g_oteZone.valid=false; if(!g_lastDisplacement.valid) return;
   double start=g_lastDisplacement.impulseStart, end=g_lastDisplacement.impulseEnd; if(start==0 || end==0 || start==end) return;
   double range=MathAbs(end-start); if(range < GetATRValue(1) * InpImpulseMinATR * 0.5) return;
   if(g_lastDisplacement.direction==1){ g_oteZone.upper=NormalizePrice(end-range*InpOTELevel1); g_oteZone.lower=NormalizePrice(end-range*InpOTELevel2); g_oteZone.direction=1; g_oteZone.valid=true; }
   else { g_oteZone.lower=NormalizePrice(end+range*InpOTELevel1); g_oteZone.upper=NormalizePrice(end+range*InpOTELevel2); g_oteZone.direction=-1; g_oteZone.valid=true; }
}

void DetermineHTFBias()
{
   int leftBars=InpSwingLeft, rightBars=InpSwingRight, pivotBar=MathMax(1,rightBars);
   if(!IsValidShift(InpBiasTF,pivotBar+leftBars) || !IsValidShift(InpBiasTF,pivotBar-rightBars)) return;
   double pivotHigh=iHigh(_Symbol,InpBiasTF,pivotBar), pivotLow=iLow(_Symbol,InpBiasTF,pivotBar); datetime pivotTime=iTime(_Symbol,InpBiasTF,pivotBar);
   bool isHigh=true, isLow=true;
   for(int i=1;i<=leftBars;i++){ if(iHigh(_Symbol,InpBiasTF,pivotBar+i)>=pivotHigh) isHigh=false; if(iLow(_Symbol,InpBiasTF,pivotBar+i)<=pivotLow) isLow=false; }
   for(int i=1;i<=rightBars;i++){ if(iHigh(_Symbol,InpBiasTF,pivotBar-i)>=pivotHigh) isHigh=false; if(iLow(_Symbol,InpBiasTF,pivotBar-i)<=pivotLow) isLow=false; }

   if(isHigh && !SwingExists(g_htfSwings,g_htfSwingCount,pivotTime,true,SWING_EXTERNAL)) AddSwingToArray(g_htfSwings,g_htfSwingCount,pivotTime,pivotHigh,true,pivotBar,SWING_EXTERNAL);
   if(isLow && !SwingExists(g_htfSwings,g_htfSwingCount,pivotTime,false,SWING_EXTERNAL)) AddSwingToArray(g_htfSwings,g_htfSwingCount,pivotTime,pivotLow,false,pivotBar,SWING_EXTERNAL);

   if(g_htfSwingCount < 4){ g_htfBias = BIAS_NONE; return; }
   SwingPoint h1,h2,l1,l2; bool gotH=GetLastTwoSwingFromArray(g_htfSwings,g_htfSwingCount,true,h1,h2), gotL=GetLastTwoSwingFromArray(g_htfSwings,g_htfSwingCount,false,l1,l2);
   if(!gotH || !gotL){ g_htfBias=BIAS_NONE; return; }
   bool HH=(h1.price>h2.price), HL=(l1.price>l2.price), LH=(h1.price<h2.price), LL=(l1.price<l2.price);
   if(HH&&HL) g_htfBias=BIAS_BULLISH; else if(LH&&LL) g_htfBias=BIAS_BEARISH; else g_htfBias=BIAS_NONE;
   WriteDebugLog("HTF bias: " + BiasToString(g_htfBias));
}

bool GetLastTwoSwingFromArray(SwingPoint &arr[], int count, bool wantHigh, SwingPoint &latest, SwingPoint &previous)
{
   int found=0; for(int i=count-1;i>=0;i--){ if(arr[i].isHigh==wantHigh){ if(found==0) latest=arr[i]; else { previous=arr[i]; return true; } found++; }} return false;
}

//============================================================
// STATE MACHINE + TRADE BUILD
//============================================================
int BarsSinceTime(datetime t){ if(t<=0) return 9999; int shift=iBarShift(_Symbol,InpSignalTF,t,true); if(shift<0) return 9999; return shift; }
bool IsContextExpired(SetupContext &ctx, int maxBars){ if(ctx.startedAt<=0) return false; return (BarsSinceTime(ctx.startedAt) > maxBars); }

void UpdateSetupContext(SetupContext &ctx)
{
   if(ctx.invalid){ ResetContextToWaitSweep(ctx); return; }

   switch(ctx.state)
   {
      case SETUP_WAIT_SWEEP:
         if(g_lastSweep.valid && g_lastSweep.direction==ctx.direction){ ctx.sweep=g_lastSweep; ctx.startedAt=g_lastSweep.time; MoveSetupState(ctx, SETUP_SWEEP_CONFIRMED); }
         break;

      case SETUP_SWEEP_CONFIRMED:
         if(IsContextExpired(ctx, InpSweepExpiryBars)){ InvalidateSetupContext(ctx, INVALID_TIMEOUT_SWEEP, "No structure shift after sweep"); break; }
         if(g_lastStructure.valid && g_lastStructure.direction==ctx.direction && g_lastStructure.time>=ctx.sweep.time)
         {
            ctx.structure=g_lastStructure;
            MoveSetupState(ctx, SETUP_SHIFT_CONFIRMED);
         }
         break;

      case SETUP_SHIFT_CONFIRMED:
         if(IsContextExpired(ctx, InpSetupExpiryBars)){ InvalidateSetupContext(ctx, INVALID_TIMEOUT_SHIFT, "No displacement after shift"); break; }
         if(g_lastDisplacement.valid && g_lastDisplacement.direction==ctx.direction && g_lastDisplacement.time>=ctx.structure.time)
         {
            ctx.displacement=g_lastDisplacement;
            if(InpUseFVG)
            {
               if(!(g_lastFVG.valid && g_lastFVG.direction==ctx.direction && g_lastFVG.timeEnd>=ctx.structure.time)) break;
               ctx.fvg=g_lastFVG;
            }
            if(InpUseOTE)
            {
               if(!(g_oteZone.valid && g_oteZone.direction==ctx.direction)) break;
               ctx.ote=g_oteZone;
            }
            MoveSetupState(ctx, SETUP_DISPLACEMENT_READY);
         }
         break;

      case SETUP_DISPLACEMENT_READY:
         BuildTradeFromContext(ctx);
         if(ctx.trade.valid) MoveSetupState(ctx, SETUP_ENTRY_READY);
         else if(IsContextExpired(ctx, InpSetupExpiryBars)) InvalidateSetupContext(ctx, INVALID_TIMEOUT_RETRACE, "No valid retracement entry");
         break;

      case SETUP_ENTRY_READY:
         if(IsContextExpired(ctx, InpSetupExpiryBars)){ InvalidateSetupContext(ctx, INVALID_TIMEOUT_RETRACE, "Entry ready timeout"); break; }
         BuildTradeFromContext(ctx);
         if(!ctx.trade.valid) InvalidateSetupContext(ctx, INVALID_RETRACE_TOO_DEEP, "Retracement no longer valid");
         break;

      case SETUP_ORDER_PLACED:
         if(!HasAnyExposure(ctx.direction)) ResetContextToWaitSweep(ctx);
         break;

      case SETUP_INVALID:
      default:
         ResetContextToWaitSweep(ctx);
         break;
   }
}

void BuildTradeSetup()
{
   g_currentSetup.valid=false;
   if(InpAllowBuy && g_buyCtx.state==SETUP_ENTRY_READY && g_buyCtx.trade.valid && (g_htfBias==BIAS_BULLISH || g_htfBias==BIAS_NONE)) g_currentSetup=g_buyCtx.trade;
   if(!g_currentSetup.valid && InpAllowSell && g_sellCtx.state==SETUP_ENTRY_READY && g_sellCtx.trade.valid && (g_htfBias==BIAS_BEARISH || g_htfBias==BIAS_NONE)) g_currentSetup=g_sellCtx.trade;
}

string BuildReasonText(SetupContext &ctx)
{
   string text = DirectionToText(ctx.direction) + " Sweep";
   text += (ctx.structure.type == STRUCT_CHOCH ? " + MSS/CHoCH" : " + BOS");
   text += " + Displacement";
   if(InpUseFVG) text += " + FVG";
   if(InpUseOTE) text += " + OTE";
   return text;
}

void BuildTradeFromContext(SetupContext &ctx)
{
   ctx.trade.valid=false;
   if(!ctx.sweep.valid || !ctx.structure.valid || !ctx.displacement.valid) return;

   double entryPrice=0; bool usePending=false;
   double ask=g_symbol.Ask(), bid=g_symbol.Bid(); double marketPrice=(ctx.direction==1?ask:bid);

   if(InpUseFVG)
   {
      if(!ctx.fvg.valid) return;
      double fvgMid=(ctx.fvg.upper + ctx.fvg.lower) * 0.5;
      if(ctx.direction==1)
      {
         if(InpEntryMode==ENTRY_MID_FVG) entryPrice=fvgMid; else if(InpEntryMode==ENTRY_NEAR_FVG) entryPrice=ctx.fvg.upper; else entryPrice=MathMin(marketPrice,fvgMid);
         bool inside=(marketPrice>=ctx.fvg.lower && marketPrice<=ctx.fvg.upper), above=(marketPrice>ctx.fvg.upper);
         if(inside){ entryPrice=marketPrice; usePending=false; } else if(above && InpAllowPending){ usePending=true; } else return;
      }
      else
      {
         if(InpEntryMode==ENTRY_MID_FVG) entryPrice=fvgMid; else if(InpEntryMode==ENTRY_NEAR_FVG) entryPrice=ctx.fvg.lower; else entryPrice=MathMax(marketPrice,fvgMid);
         bool inside=(marketPrice>=ctx.fvg.lower && marketPrice<=ctx.fvg.upper), below=(marketPrice<ctx.fvg.lower);
         if(inside){ entryPrice=marketPrice; usePending=false; } else if(below && InpAllowPending){ usePending=true; } else return;
      }
   }
   else { entryPrice=marketPrice; usePending=false; }

   if(InpUseOTE){ if(!ctx.ote.valid) return; if(entryPrice<ctx.ote.lower || entryPrice>ctx.ote.upper) return; }

   double buf=InpSLBufferPoints*_Point; double sl=0;
   if(ctx.direction==1){ sl=NormalizePrice(MathMin(ctx.sweep.sweepLow-buf, ctx.displacement.impulseStart-buf)); entryPrice=NormalizePrice(entryPrice); if(entryPrice<=sl) return; }
   else { sl=NormalizePrice(MathMax(ctx.sweep.sweepHigh+buf, ctx.displacement.impulseStart+buf)); entryPrice=NormalizePrice(entryPrice); if(entryPrice>=sl) return; }

   double riskDist=MathAbs(entryPrice-sl); if(riskDist<=0) return;
   double tp1=(ctx.direction==1?entryPrice+riskDist*InpRR_TP1:entryPrice-riskDist*InpRR_TP1);
   double tp2=(ctx.direction==1?entryPrice+riskDist*InpRR_TP2:entryPrice-riskDist*InpRR_TP2);

   ctx.trade.valid=true; ctx.trade.direction=ctx.direction; ctx.trade.signalTime=TimeCurrent(); ctx.trade.entryPrice=entryPrice; ctx.trade.stopLoss=NormalizePrice(sl);
   ctx.trade.takeProfit1=NormalizePrice(tp1); ctx.trade.takeProfit2=NormalizePrice(tp2); ctx.trade.sweepLevel=ctx.sweep.liquidityLevel;
   ctx.trade.fvgHigh=ctx.fvg.upper; ctx.trade.fvgLow=ctx.fvg.lower; ctx.trade.oteHigh=ctx.ote.upper; ctx.trade.oteLow=ctx.ote.lower;
   ctx.trade.reason=BuildReasonText(ctx); ctx.trade.execType=(usePending?EXEC_PENDING_LIMIT:EXEC_MARKET);
   ctx.trade.expiryTime=iTime(_Symbol, InpSignalTF, 0) + PeriodSeconds(InpSignalTF) * InpPendingExpiryBars;
}

//============================================================
// VALIDATION, EXECUTION, MANAGEMENT, VISUALIZATION
//============================================================
bool CheckSessionFilter(); bool HasOpenPosition(int direction); bool HasPendingOrder(int direction); bool HasAnyExposure(int direction);
bool ValidateBrokerLevels(TradeSetup &setup);
void ConfigureTradeFilling();

double NormalizeVolumeToStep(double volume)
{
   double minLot=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN), maxLot=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX), lotStep=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(lotStep<=0) lotStep=0.01;
   volume=MathMax(minLot,MathMin(maxLot,volume)); volume=MathFloor(volume/lotStep)*lotStep;
   int stepDigits=(lotStep>=1?0:(lotStep>=0.1?1:(lotStep>=0.01?2:3)));
   return NormalizeDouble(volume,stepDigits);
}

double CalculateLotSize(double entryPrice, double stopLoss)
{
   double lots=InpFixedLot;
   if(InpRiskMode==RISK_PERCENT)
   {
      double balance=AccountInfoDouble(ACCOUNT_BALANCE), riskMoney=balance*InpRiskPercent/100.0, slDist=MathAbs(entryPrice-stopLoss);
      if(slDist<=0) return 0;
      double tickValue=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE), tickSize=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
      if(tickValue<=0 || tickSize<=0) return 0;
      double lossPerLot=(slDist/tickSize)*tickValue; if(lossPerLot<=0) return 0;
      lots=riskMoney/lossPerLot;
   }
   return NormalizeVolumeToStep(lots);
}

bool ValidateTradeSetup()
{
   if(!g_currentSetup.valid) return false;
   if(!CheckSessionFilter()){ WriteDebugLog("Trade blocked: session filter"); return false; }
   long spread=g_symbol.Spread(); if(spread>InpMaxSpreadPoints){ WriteDebugLog("Trade blocked: spread too high"); return false; }
   if(InpOneTradePerSymbol && HasAnyExposure(g_currentSetup.direction)){ WriteDebugLog("Trade blocked: existing exposure"); return false; }

   if(InpUsePremiumDiscountFilter)
   {
      SwingPoint extHigh,extLow;
      if(GetLastUnbrokenSwing(SWING_EXTERNAL,true,extHigh) && GetLastUnbrokenSwing(SWING_EXTERNAL,false,extLow))
      {
         double dealingMid=(extHigh.price+extLow.price)*0.5;
         if(g_currentSetup.direction==1 && g_currentSetup.entryPrice>dealingMid) return false;
         if(g_currentSetup.direction==-1 && g_currentSetup.entryPrice<dealingMid) return false;
      }
   }
   if(!ValidateBrokerLevels(g_currentSetup)){ WriteDebugLog("Trade blocked: broker stop/freeze levels"); return false; }
   return true;
}

bool ValidateBrokerLevels(TradeSetup &setup)
{
   double stopsLevel=g_symbol.StopsLevel()*_Point, freezeLevel=g_symbol.FreezeLevel()*_Point, ask=g_symbol.Ask(), bid=g_symbol.Bid();
   if(setup.execType==EXEC_MARKET)
   {
      if(setup.direction==1){ if((ask-setup.stopLoss)<stopsLevel) return false; }
      else { if((setup.stopLoss-bid)<stopsLevel) return false; }
   }
   else
   {
      if(setup.direction==1){ if((ask-setup.entryPrice)>0 && (ask-setup.entryPrice)<freezeLevel) return false; if((setup.entryPrice-setup.stopLoss)<stopsLevel) return false; }
      else { if((setup.entryPrice-bid)>0 && (setup.entryPrice-bid)<freezeLevel) return false; if((setup.stopLoss-setup.entryPrice)<stopsLevel) return false; }
   }
   return true;
}

void ExecuteTradeSetup(TradeSetup &setup)
{
   ConfigureTradeFilling();
   bool sent=false; string comment=InpComment + "_V3";
   if(setup.execType==EXEC_MARKET)
   {
      double price=(setup.direction==1?g_symbol.Ask():g_symbol.Bid());
      if(setup.direction==1) sent=g_trade.Buy(setup.lotSize,_Symbol,price,setup.stopLoss,setup.takeProfit2,comment);
      else sent=g_trade.Sell(setup.lotSize,_Symbol,price,setup.stopLoss,setup.takeProfit2,comment);
   }
   else
   {
      if(setup.direction==1) sent=g_trade.BuyLimit(setup.lotSize,setup.entryPrice,_Symbol,setup.stopLoss,setup.takeProfit2,ORDER_TIME_SPECIFIED,setup.expiryTime,comment);
      else sent=g_trade.SellLimit(setup.lotSize,setup.entryPrice,_Symbol,setup.stopLoss,setup.takeProfit2,ORDER_TIME_SPECIFIED,setup.expiryTime,comment);
   }

   if(sent)
   {
      ulong orderTicket = g_trade.ResultOrder();
      if(setup.direction==1){ g_buyCtx.orderTicket = orderTicket; MoveSetupState(g_buyCtx, SETUP_ORDER_PLACED); }
      else { g_sellCtx.orderTicket = orderTicket; MoveSetupState(g_sellCtx, SETUP_ORDER_PLACED); }

      g_positionRuntime.ticket=0; g_positionRuntime.initialVolume=setup.lotSize; g_positionRuntime.partialDone=false;
      WriteDebugLog("Order placed: " + setup.reason + ", ticket=" + IntegerToString((int)orderTicket));
      g_currentSetup.valid=false;
   }
   else WriteDebugLog("Order failed. Retcode=" + IntegerToString((int)g_trade.ResultRetcode()) + " " + g_trade.ResultRetcodeDescription());
}

void ConfigureTradeFilling()
{
   long fillingMode=0;
   if(SymbolInfoInteger(_Symbol,SYMBOL_FILLING_MODE,fillingMode))
   {
      if((fillingMode & SYMBOL_FILLING_FOK)==SYMBOL_FILLING_FOK) g_trade.SetTypeFilling(ORDER_FILLING_FOK);
      else if((fillingMode & SYMBOL_FILLING_IOC)==SYMBOL_FILLING_IOC) g_trade.SetTypeFilling(ORDER_FILLING_IOC);
      else g_trade.SetTypeFilling(ORDER_FILLING_RETURN);
   }
   else g_trade.SetTypeFilling(ORDER_FILLING_RETURN);
}

void ApplyATRTrailing(ulong ticket, int dir, double currentPrice, double openPrice, double sl, double tp)
{ double atr=GetATRValue(0); if(atr<=0) return; double trailDist=atr*InpTrailingATRMult, newSL=sl; if(dir==1){ double candidate=NormalizePrice(currentPrice-trailDist); if(candidate>sl && candidate>openPrice) newSL=candidate; } else { double candidate=NormalizePrice(currentPrice+trailDist); if(candidate<sl && candidate<openPrice) newSL=candidate; } if(newSL!=sl) g_trade.PositionModify(ticket,newSL,tp); }
void ApplySwingTrailing(ulong ticket, int dir, double sl, double tp)
{ SwingPoint sw; if(!GetLastUnbrokenSwing(SWING_INTERNAL, dir==1?false:true, sw)) return; double buf=InpSLBufferPoints*_Point, newSL=sl; if(dir==1){ double c=NormalizePrice(sw.price-buf); if(c>sl) newSL=c; } else { double c=NormalizePrice(sw.price+buf); if(c<sl) newSL=c; } if(newSL!=sl) g_trade.PositionModify(ticket,newSL,tp); }

void ManagePositions()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      if(!g_position.SelectByIndex(i)) continue;
      if(g_position.Symbol()!=_Symbol || g_position.Magic()!=InpMagicNumber) continue;
      ulong ticket=g_position.Ticket(); double openPrice=g_position.PriceOpen(), sl=g_position.StopLoss(), tp=g_position.TakeProfit(), volume=g_position.Volume(); int dir=(g_position.PositionType()==POSITION_TYPE_BUY?1:-1); double currentPrice=(dir==1?g_symbol.Bid():g_symbol.Ask());
      if(g_positionRuntime.ticket!=ticket){ g_positionRuntime.ticket=ticket; g_positionRuntime.initialVolume=volume; g_positionRuntime.partialDone=false; }
      double initialRisk=MathAbs(openPrice-sl); if(initialRisk<=0) continue;
      double tp1=(dir==1?openPrice+initialRisk*InpRR_TP1:openPrice-initialRisk*InpRR_TP1); bool tp1Hit=(dir==1?currentPrice>=tp1:currentPrice<=tp1);
      if(InpUsePartialClose && !g_positionRuntime.partialDone && tp1Hit)
      {
         double closeVolume=NormalizeVolumeToStep(volume * InpPartialPercent / 100.0);
         if(closeVolume>0 && closeVolume<volume && g_trade.PositionClosePartial(ticket, closeVolume))
         {
            g_positionRuntime.partialDone=true;
            if(InpMoveToBEAfterTP1)
            {
               double bePrice=(dir==1?openPrice+InpBESafetyPoints*_Point:openPrice-InpBESafetyPoints*_Point);
               bool improve=(dir==1?bePrice>sl:bePrice<sl);
               if(improve) g_trade.PositionModify(ticket, NormalizePrice(bePrice), tp);
            }
         }
      }
      if(InpUseTrailing){ if(InpTrailingMode==TRAIL_ATR) ApplyATRTrailing(ticket,dir,currentPrice,openPrice,sl,tp); else ApplySwingTrailing(ticket,dir,sl,tp); }
   }
}

void ManagePendingOrders()
{
   for(int i=OrdersTotal()-1;i>=0;i--)
   {
      ulong ticket=OrderGetTicket(i); if(ticket==0) continue; if(!OrderSelect(ticket)) continue;
      string symbol=OrderGetString(ORDER_SYMBOL); long magic=OrderGetInteger(ORDER_MAGIC); if(symbol!=_Symbol || magic!=InpMagicNumber) continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE); if(type!=ORDER_TYPE_BUY_LIMIT && type!=ORDER_TYPE_SELL_LIMIT) continue;
      datetime expiration=(datetime)OrderGetInteger(ORDER_TIME_EXPIRATION);
      if(expiration>0 && TimeCurrent()>expiration)
      {
         MqlTradeRequest req; MqlTradeResult res; ZeroMemory(req); ZeroMemory(res); req.action=TRADE_ACTION_REMOVE; req.order=ticket;
         if(!OrderSend(req,res)) WriteDebugLog("Failed remove expired order ticket=" + IntegerToString((int)ticket));
      }
   }
}

void ManageOrdersAndPositions(){ g_symbol.RefreshRates(); ManagePositions(); ManagePendingOrders(); }

bool CheckSessionFilter()
{
   if(!InpUseSessionFilter) return true;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt); int h=dt.hour;
   bool inLondon=(h>=InpLondonStart && h<InpLondonEnd), inNewYork=(h>=InpNewYorkStart && h<InpNewYorkEnd);
   if(InpTradeSession==SESSION_LONDON) return inLondon;
   if(InpTradeSession==SESSION_NEWYORK) return inNewYork;
   return (inLondon || inNewYork);
}

bool HasOpenPosition(int direction){ for(int i=PositionsTotal()-1;i>=0;i--){ if(!g_position.SelectByIndex(i)) continue; if(g_position.Symbol()!=_Symbol || g_position.Magic()!=InpMagicNumber) continue; int posDir=(g_position.PositionType()==POSITION_TYPE_BUY?1:-1); if(posDir==direction) return true; } return false; }
bool HasPendingOrder(int direction){ for(int i=OrdersTotal()-1;i>=0;i--){ ulong t=OrderGetTicket(i); if(t==0 || !OrderSelect(t)) continue; if(OrderGetString(ORDER_SYMBOL)!=_Symbol || OrderGetInteger(ORDER_MAGIC)!=InpMagicNumber) continue; ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE); if(direction==1 && type==ORDER_TYPE_BUY_LIMIT) return true; if(direction==-1 && type==ORDER_TYPE_SELL_LIMIT) return true; } return false; }
bool HasAnyExposure(int direction){ return (HasOpenPosition(direction) || HasPendingOrder(direction)); }

void EnsureArrow(string name, datetime t, double price, int code, color c){ if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_ARROW,0,t,price); ObjectSetInteger(0,name,OBJPROP_TIME,0,t); ObjectSetDouble(0,name,OBJPROP_PRICE,0,price); ObjectSetInteger(0,name,OBJPROP_ARROWCODE,code); ObjectSetInteger(0,name,OBJPROP_COLOR,c); ObjectSetInteger(0,name,OBJPROP_WIDTH,1); }
void EnsureText(string name, datetime t, double price, string text){ if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_TEXT,0,t,price); ObjectSetInteger(0,name,OBJPROP_TIME,0,t); ObjectSetDouble(0,name,OBJPROP_PRICE,0,price); ObjectSetString(0,name,OBJPROP_TEXT,text); ObjectSetInteger(0,name,OBJPROP_COLOR,clrWhite); ObjectSetInteger(0,name,OBJPROP_FONTSIZE,8); }
void EnsureHLine(string name,double price,color c,ENUM_LINE_STYLE style){ if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_HLINE,0,0,price); ObjectSetDouble(0,name,OBJPROP_PRICE,price); ObjectSetInteger(0,name,OBJPROP_COLOR,c); ObjectSetInteger(0,name,OBJPROP_STYLE,style); ObjectSetInteger(0,name,OBJPROP_WIDTH,1); }
void EnsureRectangle(string name, datetime t1,double p1,datetime t2,double p2,color c){ if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_RECTANGLE,0,t1,p1,t2,p2); ObjectSetInteger(0,name,OBJPROP_TIME,0,t1); ObjectSetDouble(0,name,OBJPROP_PRICE,0,p1); ObjectSetInteger(0,name,OBJPROP_TIME,1,t2); ObjectSetDouble(0,name,OBJPROP_PRICE,1,p2); ObjectSetInteger(0,name,OBJPROP_COLOR,c); ObjectSetInteger(0,name,OBJPROP_BACK,true); ObjectSetInteger(0,name,OBJPROP_FILL,true); ObjectSetInteger(0,name,OBJPROP_WIDTH,1); }
void DrawSwingArray(SwingPoint &arr[], int count, string prefix, color defaultColor){ int startIdx=MathMax(0,count-12); for(int i=startIdx;i<count;i++){ string nm=g_objPrefix+prefix+IntegerToString(i); color c=defaultColor; if(arr[i].isHigh) c=(arr[i].broken?clrDimGray:clrTomato); else c=(arr[i].broken?clrDimGray:clrDodgerBlue); EnsureArrow(nm,arr[i].time,arr[i].price,(arr[i].isHigh?218:217),c);} }

void DrawICTObjects()
{
   if(InpShowSwings){ DrawSwingArray(g_internalSwings,g_internalCount,"ISW_",clrSilver); DrawSwingArray(g_externalSwings,g_externalCount,"ESW_",clrWhite); }
   if(InpShowBOSCHoCH && g_lastStructure.valid){ string nm=g_objPrefix+"STRUCT"; EnsureHLine(nm,g_lastStructure.level,(g_lastStructure.direction==1?clrDodgerBlue:clrOrangeRed),STYLE_DASH); EnsureText(g_objPrefix+"STRUCT_TXT", g_lastStructure.time, g_lastStructure.level, StructureTypeToString(g_lastStructure.type)); }
   if(InpShowSweep && g_lastSweep.valid){ double px=(g_lastSweep.direction==1?g_lastSweep.sweepLow:g_lastSweep.sweepHigh); EnsureArrow(g_objPrefix+"SWEEP",g_lastSweep.time,px,241,clrYellow); EnsureText(g_objPrefix+"SWEEP_TXT",g_lastSweep.time,px,(g_lastSweep.direction==1?"SSL Sweep":"BSL Sweep")); }
   if(InpShowFVG && g_lastFVG.valid){ datetime t2=iTime(_Symbol,InpSignalTF,0)+PeriodSeconds(InpSignalTF)*8; EnsureRectangle(g_objPrefix+"FVG",g_lastFVG.timeStart,g_lastFVG.upper,t2,g_lastFVG.lower,(g_lastFVG.direction==1?clrLightBlue:clrLightCoral)); }
   if(InpShowOTE && g_oteZone.valid){ datetime t2=iTime(_Symbol,InpSignalTF,0)+PeriodSeconds(InpSignalTF)*6; EnsureRectangle(g_objPrefix+"OTE",iTime(_Symbol,InpSignalTF,1),g_oteZone.upper,t2,g_oteZone.lower,clrGold); }
   if(InpShowEntry && g_currentSetup.valid){ EnsureHLine(g_objPrefix+"ENTRY",g_currentSetup.entryPrice,clrLime,STYLE_DOT); EnsureHLine(g_objPrefix+"SL",g_currentSetup.stopLoss,clrRed,STYLE_DOT); EnsureHLine(g_objPrefix+"TP1",g_currentSetup.takeProfit1,clrLime,STYLE_DOT); EnsureHLine(g_objPrefix+"TP2",g_currentSetup.takeProfit2,clrLime,STYLE_DOT); }
}

void DeleteAllICTObjects(){ int total=ObjectsTotal(0); for(int i=total-1;i>=0;i--){ string nm=ObjectName(0,i); if(StringFind(nm,g_objPrefix)==0) ObjectDelete(0,nm); } }
//+------------------------------------------------------------------+

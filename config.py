"""Tunable settings for the NFL edge model. Every number here is a modelling
assumption — change them, then re-run backtest.py to see whether calibration improves."""
import os

# ---- data ---------------------------------------------------------------
CACHE_DIR = os.environ.get("NFL_EDGE_CACHE", os.path.join(os.path.dirname(__file__), "cache"))
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")          # https://the-odds-api.com (free tier: 500 credits/mo)
BOOKMAKER = "fanduel"

# ---- blending last season with this season -----------------------------
# Prior-season plays get weight K / (K + games_played_this_season).
# K=2 (tuned on 2024+2025 weeks 2-5, confirmed neutral in weeks 12-18): after 2 games a prior-season
# play counts 0.67 of a current one, after 8 games 0.20. Lower K adapts faster to role changes
# (a player whose usage jumped this season) and beat K=4/6/12 on every market early in the year.
PRIOR_SEASON_K = 2.0
# Recency half-life (in games) for player usage shares.
USAGE_HALF_LIFE_GAMES = 4.0
# Injury-shortened / snap-limited games (a regular playing < 60% of his usual snaps):
#   "drop"  -> ignore them;  "scale" -> keep them, scale shares up by (usual/actual snaps)^ALPHA
#   (capped 2.5x) and down-weight the game by actual/usual snaps.
SHORT_GAME_MODE = "scale"
# Only treat a low-snap game as injury-shortened if the player was on that week's injury report
# (or left the game hurt). Otherwise a low snap count is simply that player's role.
SHORT_GAME_NEEDS_INJURY = True
SHORT_GAME_ALPHA = 0.5
# Games with a player's PREVIOUS team count at this weight (role info for new arrivals).
PRIOR_TEAM_WEIGHT = 0.5
# Shares shrink toward a position prior with this many pseudo-games (tames one-game samples).
SHARE_PRIOR_GAMES = 1.0
# Per-position shrinkage. QB carry share is highly individual (a runner vs a pocket passer), so
# shrinking it toward a league prior compressed rushing projections badly: pocket QBs were
# projected 13.9 yds against an actual 1.7, runners 27.5 against an actual 45.8 (2025).
SHARE_PRIOR_GAMES_BY_POS = {}   # tested QB=0.15: made QB rushing worse, reverted
SHARE_PRIORS = {"WR": (0.10, 0.0), "TE": (0.08, 0.0), "RB": (0.06, 0.25), "QB": (0.0, 0.06)}  # (target, carry)

# ---- empirical-Bayes shrinkage (pseudo-counts toward position mean) -----
SHRINK_TARGETS = 60        # yards/target, catch rate
SHRINK_CARRIES = 80        # yards/carry
SHRINK_DEF_PLAYS = 250     # defensive EPA / yards allowed
SHRINK_COVERAGE_TGTS = 40  # receiver man/zone splits

# ---- simulation ---------------------------------------------------------
# Values below were tuned by tune.py (coordinate descent on Brier score, 2025 weeks 4-11)
# and confirmed on held-out 2025 weeks 12-18. Re-tune each off-season.
N_SIMS = 20000
MARGIN_SD = 13.5           # SD of NFL final margin vs spread (historical ~13-14)
TOTAL_SD = 10.0            # SD of game total vs closing total
PLAYS_SD = 5.5             # SD of a team's offensive plays per game
SCRIPT_PASS_BETA = 0.03   # pass-rate change per 14 pts of (opponent - team) margin
FUNNEL_BETA = 0.01        # pass-rate change per 1 SD of defensive pass/run funnel
POINTS_PER_TD = 8.1        # team TDs ~ Poisson(points / 7.3)
YPR_GAMMA_SHAPE = 1.35     # yards-per-reception gamma shape (right-skewed)
RUSH_SD_PER_CARRY = 7.0    # per-carry yardage SD
RUSH_T_DF = 4              # Student-t df for rushing yardage tail
EFFICIENCY_POINTS_ELASTICITY = 0.5
# Dirichlet concentration for per-game usage shares (lower = more game-to-game volatility).
# Without this, a fixed-share multinomial is overconfident about player volume.
TARGET_SHARE_CONC = 70
CARRY_SHARE_CONC = 12
CATCH_RATE_SD = 0.06       # per-game noise in a receiver's catch rate  # yards scale with (sim points / expected points)^e

# ---- betting ------------------------------------------------------------
# Books are sharper than any simple model. Where a two-way market exists we blend:
# p_final = MODEL_WEIGHT * p_model + (1 - MODEL_WEIGHT) * p_market_fair
MODEL_WEIGHT = 0.35
ONE_SIDED_MARGIN = {"anytime_td": 0.08, "alt": 0.06}  # assumed vig when no opposing side is posted
# Heavy chalk is a poor single bet however likely it is: -800 risks 8 units to win 1. Singles
# shorter than this are hidden (parlay legs are exempt — combining them is what pays).
MIN_PRICE_AMERICAN = -350
MIN_EDGE = 0.03            # flag only if blended prob beats break-even by >= 3 pts
KELLY_FRACTION = 0.25
KELLY_CAP = 0.02           # never suggest more than 2% of bankroll on one bet

# Markets the model is NOT trusted on (backtest: QB passing yards over-projects by ~20 yds,
# lowest skill of any market). Excluded from best_bets() until fixed.
EXCLUDE_STATS = {"pass_yds"}

# ---- weather (estimated by weather_study.py on 2021-25, controlling for the closing total and
# team-season offense/defense effects; only effects with |t| > 2 are used). Books already lower
# totals for weather, so these shift pass/run MIX and passing efficiency, never total scoring.
WX_COLD_PASS = -0.030      # neutral pass rate, outdoor games below 32F        (t = -3.3)
WX_PRECIP_PASS = -0.018    # neutral pass rate, rain/snow                      (t = -2.3)
WX_PRECIP_CMP = -0.026     # catch rate, rain/snow                             (t = -3.6)
WX_PRECIP_YPT = -0.53      # yards per target, rain/snow                       (t = -3.6)
WX_WIND_YPT = -0.019       # yards per target per mph of wind                  (t = -3.1)

# ---- questionable players: simulated as a coin flip on whether they play (teammates absorb
# the volume in the scenarios where they sit). Base rate from 2024-25 injury reports.
P_PLAY_QUESTIONABLE = 0.56   # measured: 792 skill-position cases, 2024-25
MIXTURE_CHUNKS = 40
USE_WEATHER = True         # set False to switch weather adjustments off

# ---- defensive modelling switches (see defense.py; each validated in backtest before enabling)
USE_OPP_ADJ = False  # opponent-adjusted defensive ratings instead of raw averages
USE_DVP = False      # defense-vs-position: target share and yards/target allowed to WR/TE/RB
USE_SHELL = False    # two-high vs single-high shell rate x each receiver's shell split
DEF_STRENGTH = 1.0   # 0..1: how strongly the defensive features above are applied

# ---- QB early exit: in 12.4% of 2025 starts (55/445) the starter threw < 85% of team attempts
# (injury, benching, blowout). Books settle QB props if he plays a snap, so this risk belongs in
# the price. When it happens he keeps a Uniform(QB_EXIT_FRAC) share of the team's passing.
P_QB_EXIT = 0.0         # measured 0.124, but OFF: no Brier gain (calibration already absorbs it); see README
QB_EXIT_FRAC = (0.05, 0.75)

# ---- game outcome distribution: "empirical" = historical games with similar spread/total
# (keeps key numbers 3/7/10 and margin-total correlation); "normal" = bell curve.
# Empirical beat normal on ML and alt-spread Brier out of sample (2021-25, trained 2003-20).
MARGIN_MODEL = "empirical"
TEASER_DOG_RANGE = (1.5, 2.5)     # 6-pt teaser legs that still clear breakeven in 2021-25 (76.5%)

# QB rushing: the simulation compresses QBs toward the average (pocket passers projected 13.9 yds
# vs 1.7 actual; runners 27.5 vs 45.8, 2025). A QB's own recent rushing average was more accurate
# (MAE 12.7 vs 13.4, bias -0.1 vs +3.8), so the QB's rushing is blended toward it.
QB_RUSH_TRAIL_WEIGHT = 0.75   # tuned: best Brier, QB rush MAE 13.55 -> 12.43

# ---- defensive injuries: a defense missing starters gives up more.
# Sized by judgement, NOT measured — the multiplier is applied to how many yards that defense
# allows, scaled by the share of its normal defensive snaps that are out. "1.0 missing" means a
# full-time starter's worth of snaps. Validate against past seasons before trusting it heavily.
AUTO_DEF_ADJUST = True
DEF_INJURY_DB_YPT = 0.05        # +5% yards per target per starter-equivalent DB missing
DEF_INJURY_DB_CMP = 0.015       # +1.5% completion rate likewise
DEF_INJURY_FRONT_YPC = 0.045    # +4.5% yards per carry per starter-equivalent front-seven missing
DEF_INJURY_FRONT_YPT = 0.015    # pass rush matters for the pass game too, a little
DEF_INJURY_CAP = 0.15           # never swing a defense by more than 15%

# Red-zone and third-down defense. Used only to shape HOW a team scores (pass vs run touchdowns)
# and how many plays a game holds — never how much it scores, since the market total covers that.
USE_REDZONE = True
RZ_TD_SPLIT_BETA = 0.5      # how strongly a defense's pass/run TD profile shifts the split
THIRD_DOWN_PLAYS_BETA = 4.0 # extra plays per game when a defense is poor on third down

# ---- red-zone defence: shifts the TD/FG mix without changing total points.
# Tested on 2025 (both halves, 5,288 player-games): anytime-TD Brier -0.00017, better in each half
# independently, 95% CI [-0.00038, +0.00002] — consistent but at the edge of significance. ON at
# full strength; the mechanism is sound and it never hurt either half.
USE_RZ_DEFENSE = True
RZ_STRENGTH = 1.0            # 1.0 = apply the opponent's full red-zone factor

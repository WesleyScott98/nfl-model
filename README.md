# Rob's NFL Model — the website

A single static page that rebuilds itself. GitHub Actions runs the model on a schedule, writes a
fresh `site/index.html`, and your host publishes it. No server to babysit.

---

## Setup, once (about 15 minutes)

### 1. Put this folder in a new GitHub repo
Create a repo called `nfl-model` (public — free Actions minutes). Upload everything in this folder,
keeping the structure:

```
build_site.py        template.html        requirements.txt
model/               the betting model
site/                the published page (rebuilt automatically)
.github/workflows/   the schedule
netlify/functions/   the "ask" proxy (only needed for written answers)
```

### 2. Turn on the schedule
Nothing to do — `.github/workflows/update-picks.yml` runs automatically once it's in the repo:

| When (UTC) | Why |
|---|---|
| Wed 15:00 | new week's lines are posted |
| Sat 15:00 | injury designations are final |
| Sun 15:00 | 11am ET, inactives landing |
| Mon & Thu 21:00 | before the standalone games |

You can also run it any time: **Actions → Update picks → Run workflow**.

### 3. Publish it — pick one

**GitHub Pages** (simplest, free forever, no written answers)
Settings → Pages → Source: *Deploy from a branch* → Branch: `main`, folder: `/site` → Save.
Live at `https://<your-username>.github.io/nfl-model`.

**Netlify** (free, and the question box gets written answers)
1. netlify.com → Add new site → Import from GitHub → pick this repo. `netlify.toml` handles the rest.
2. Site settings → Environment variables → add `ANTHROPIC_API_KEY` (from console.anthropic.com).
3. Done. Live at `https://<site-name>.netlify.app`.

Without the key the site still works — the question box answers from the page's own data, which is
how it already behaves inside Claude when the AI layer isn't available.

### 4. Live FanDuel prices (optional but worth it)
Get a free key at **the-odds-api.com**, then add it to the repo:
**Settings → Secrets and variables → Actions → New repository secret → `ODDS_API_KEY`**.

With it, each build pulls FanDuel's real spread, total, moneyline and player props. Cards then show
**FanDuel's price next to the model's fair price**, a **VALUE** badge appears wherever FanDuel is
paying more than the bet is worth, and a **Best prices** tab lists those bets ranked by edge.
The model also anchors on FanDuel's own spread and total instead of the schedule's opener.

**Credit budget** (free tier is 500/month):

| Build | Mode | Credits |
|---|---|---|
| Wed, Sat | `lines` | 3 each |
| Sun 12:45pm ET, Sun 7pm ET, Mon, Thu | `full` | ~67 each |

That's roughly 280 a month — inside the free tier with room to spare. The workflow already picks
the mode per run. If the key is missing or the call fails, the build carries on with model prices
and the site still works.

### 5. A custom domain (optional, ~$12/year)
Buy at Cloudflare or Namecheap, then follow your host's "Add custom domain" flow.

---

## Injuries between games — handled automatically
Official designations publish **Wednesday**, so a Monday-night injury is invisible to the feed for
two days. Every build closes that gap itself: any player whose snap share collapsed last week is
treated as **questionable (50% to play)** until the report covers him, at which point the report
takes over and the flag clears on its own. The build log prints who was flagged.

Week 3 caught Jaxson Dart (12% of snaps), D.J. Moore (31%), Jayden Reed (3%), Saquon Barkley (16%)
and Dallas Goedert (25%) with no input from you.

It's deliberately a 50/50 downgrade rather than ruling someone out, because a low snap count can
also mean a blowout or a rotation. If you *know* better, `overrides.json` wins:

```json
{ "out": ["Jaxson Dart"], "questionable": {"D.J. Moore": 0.6}, "snap_limit": {"Player Name": 0.5} }
```

You never *have* to touch it — it's there for when you know something the data doesn't.

## What updates on its own
- **The week itself.** The build picks whichever week's games are still ahead: Week 3 through
  Monday night, Week 4 from Tuesday. The page title, header and picks all follow. No edits.
- **Injuries**, from the official report, plus the automatic snap-collapse flag above
- Injuries, inactives, and roster moves (IR, free agents, suspensions, practice squad)
- Every player's usage as the season goes on
- Weather for outdoor games

## What does NOT update on its own
- **Nothing, if you add an odds key** (see below). Without one, the page shows the model's fair
  price only — what a bet is worth, not what FanDuel is paying.
- **The model itself.** Improvements still come from Claude; drop new files into `model/`.
- **Judgment.** Nobody reviews the output before it publishes. Feed errors go live. Glance at it
  after the Sunday build — that's how the Joe Mixon free-agent bug got caught.

---

## Running it yourself

```bash
pip install -r requirements.txt
python build_site.py                 # current week, auto-detected
python build_site.py --week 5        # a specific week
open site/index.html
```

Takes about 30 seconds for a full slate.

## Costs
- GitHub Actions: free on public repos
- GitHub Pages / Netlify hosting: free
- Written answers via the Anthropic API: a fraction of a cent per question. The function throttles
  to 20 questions a minute so a shared link can't run up a bill.

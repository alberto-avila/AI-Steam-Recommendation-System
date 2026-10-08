# Verbalizer checkpoint: 10 histories for review

Strategy: top_hours_summary; up to 15 games. Source: original training rows only.
Samples span training-history sizes, with seed-derived tie ordering. They were not chosen by model scores.
Target names below are audit labels outside the prompt. No scores or target hours enter the prompt.

Token contract: one BOS + plain history; no EOS, chat template, padding, or token truncation.
Token budget: 512. Prompts exceeding it are flagged, never silently truncated.

| Split | Users | Median tokens | p99 tokens | Max tokens | Over budget | Histories capped by K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 2436 | 147 | 230 | 277 | 0 | 886 |
| test | 2436 | 158 | 231 | 277 | 0 | 940 |

## 1. User 53258793

Training games: 4; rendered: 4; omitted by K: 0; input tokens: 83.
Held-out test label (excluded): **Left 4 Dead 2** (ID 1703).

```text
This player has played 4 games for 1521.1 hrs in total.
Their most-played game accounts for 99% of their playtime.
Most-played games:
Counter-Strike (1511 hrs), Counter-Strike Condition Zero (7.4 hrs), Zombie Panic Source (1.4 hrs), Day of Defeat (1.3 hrs).
```

## 2. User 92904111

Training games: 4; rendered: 4; omitted by K: 0; input tokens: 81.
Held-out test label (excluded): **Day of Defeat** (ID 751).

```text
This player has played 4 games for 21.6 hrs in total.
Their most-played game accounts for 83% of their playtime.
Most-played games:
Empire Total War (18 hrs), Total War SHOGUN 2 (3.1 hrs), Borderlands 2 (0.3 hrs), Path of Exile (0.2 hrs).
```

## 3. User 153821154

Training games: 6; rendered: 6; omitted by K: 0; input tokens: 108.
Held-out test label (excluded): **Scribblenauts Unlimited** (ID 2565).

```text
This player has played 6 games for 84.9 hrs in total.
Their most-played game accounts for 52% of their playtime.
Most-played games:
Assassin's Creed II (44 hrs), Portal 2 (19.8 hrs), Batman Arkham Asylum GOTY Edition (16.1 hrs), The Elder Scrolls V Skyrim (2.9 hrs), Orcs Must Die! 2 (1.5 hrs), Antichamber (0.6 hrs).
```

## 4. User 67965357

Training games: 7; rendered: 7; omitted by K: 0; input tokens: 108.
Held-out test label (excluded): **Clicker Heroes** (ID 581).

```text
This player has played 7 games for 2375.4 hrs in total.
Their most-played game accounts for 98% of their playtime.
Most-played games:
Dota 2 (2329 hrs), Archeblade (22 hrs), Left 4 Dead 2 (13.6 hrs), Team Fortress 2 (6.8 hrs), Unturned (3.2 hrs), Grand Chase (0.4 hrs), Ragnarok (0.4 hrs).
```

## 5. User 59756932

Training games: 10; rendered: 10; omitted by K: 0; input tokens: 141.
Held-out test label (excluded): **Gang Beasts** (ID 1284).

```text
This player has played 10 games for 293.4 hrs in total.
Their most-played game accounts for 70% of their playtime.
Most-played games:
Dota 2 (205 hrs), Mafia II (52 hrs), PAYDAY 2 (13.3 hrs), The Walking Dead (6 hrs), Age of Mythology Extended Edition (4.5 hrs), Warframe (3.2 hrs), Banished (2.8 hrs), Left 4 Dead 2 (2.5 hrs), Garry's Mod (2.2 hrs), Rising Storm/Red Orchestra 2 Multiplayer (1.9 hrs).
```

## 6. User 242523654

Training games: 13; rendered: 13; omitted by K: 0; input tokens: 181.
Held-out test label (excluded): **Happy Wars** (ID 1438).

```text
This player has played 13 games for 69.2 hrs in total.
Their most-played game accounts for 39% of their playtime.
Most-played games:
Survival Postapocalypse Now (27 hrs), Counter-Strike Global Offensive (26 hrs), PAYDAY 2 (5.4 hrs), Just Cause 2 Multiplayer Mod (2.7 hrs), Crusader Kings II (2.4 hrs), Mount & Blade (1.7 hrs), Age of Empires III Complete Collection (1 hrs), Saints Row The Third (1 hrs), Age of Empires II HD Edition (0.7 hrs), Dead Island Riptide (0.5 hrs), Unturned (0.5 hrs), Warhammer 40,000 Dawn of War Dark Crusade (0.2 hrs), Just Cause 2 (0.1 hrs).
```

## 7. User 26027937

Training games: 18; rendered: 15; omitted by K: 3; input tokens: 199.
Held-out test label (excluded): **Team Fortress 2** (ID 2957).

```text
This player has played 18 games for 493 hrs in total.
Their most-played game accounts for 47% of their playtime.
Most-played games (top 15 of 18):
Counter-Strike Global Offensive (233 hrs), Counter-Strike Source (127 hrs), Worms Revolution (26 hrs), Mark of the Ninja (25 hrs), The Book of Unwritten Tales The Critter Chronicles (13 hrs), The Basement Collection (12 hrs), Dota 2 (11.6 hrs), Democracy 3 (10.7 hrs), LIMBO (9.4 hrs), Stealth Bastard Deluxe (8.4 hrs), Zeno Clash (5.8 hrs), Audiosurf (4.7 hrs), Natural Selection 2 (2.3 hrs), Total War SHOGUN 2 (1.7 hrs), Toki Tori (1.5 hrs).
```

## 8. User 61632730

Training games: 28; rendered: 15; omitted by K: 13; input tokens: 208.
Held-out test label (excluded): **Path of Exile** (ID 2167).

```text
This player has played 28 games for 1604.9 hrs in total.
Their most-played game accounts for 29% of their playtime.
Most-played games (top 15 of 28):
War Thunder (458 hrs), H1Z1 (418 hrs), Call of Duty Modern Warfare 2 - Multiplayer (194 hrs), Dead Island Epidemic (126 hrs), Don't Starve (109 hrs), Counter-Strike Global Offensive (108 hrs), Darkest Dungeon (63 hrs), Don't Starve Together Beta (42 hrs), Clicker Heroes (27 hrs), Echo of Soul (16.8 hrs), Infinite Crisis (13.6 hrs), Arma 2 DayZ Mod (12.1 hrs), Enclave (3.3 hrs), Dota 2 (2.5 hrs), Call of Duty Modern Warfare 2 (2.1 hrs).
```

## 9. User 182399789

Training games: 51; rendered: 15; omitted by K: 36; input tokens: 214.
Held-out test label (excluded): **Insurgency** (ID 1574).

```text
This player has played 51 games for 1395.6 hrs in total.
Their most-played game accounts for 43% of their playtime.
Most-played games (top 15 of 51):
Counter-Strike Global Offensive (601 hrs), Unturned (305 hrs), Tom Clancy's Ghost Recon Phantoms - NA (72 hrs), Arma 2 Operation Arrowhead (60 hrs), Team Fortress 2 (53 hrs), PlanetSide 2 (42 hrs), Valkyria Chronicles (22 hrs), Arma 2 (19.4 hrs), Metro 2033 (18.2 hrs), APB Reloaded (16.7 hrs), Garry's Mod (16.5 hrs), War Thunder (15.3 hrs), East India Company Gold (11.6 hrs), Penguins Arena Sedna's World (10.4 hrs), Heroes & Generals (10.3 hrs).
```

## 10. User 62990992

Training games: 497; rendered: 15; omitted by K: 482; input tokens: 221.
Held-out test label (excluded): **Warframe** (ID 3390).

```text
This player has played 497 games for 5714.3 hrs in total.
Their most-played game accounts for 12% of their playtime.
Most-played games (top 15 of 497):
Counter-Strike Global Offensive (663 hrs), Sid Meier's Civilization V (550 hrs), Total War SHOGUN 2 (212 hrs), Total War ROME II - Emperor Edition (198 hrs), Dungeon Defenders (195 hrs), Age of Empires Online (168 hrs), XCOM Enemy Unknown (126 hrs), Empire Total War (125 hrs), Might & Magic Heroes VI (118 hrs), Assassin's Creed IV Black Flag (94 hrs), Alien Swarm (83 hrs), Assassin's Creed II (79 hrs), Assassin's Creed Brotherhood (73 hrs), Terra Incognita ~ Chapter One The Descendant (61 hrs), Warlock - Master of the Arcane (55 hrs).
```

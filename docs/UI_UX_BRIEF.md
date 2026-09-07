# Review Sense — UI/UX Design Brief

## 1. Design Philosophy & Vision

The **Review Sense** dashboard is designed to provide instant, visually intuitive emotion intelligence for text analysis. 

Rather than overwhelming the user with raw numbers, the interface uses a **Color-Coded Emotion Taxonomy**, **Hero Badges**, and **Animated Probability Distribution Bars** to translate complex machine learning multi-label outputs into immediate actionable insight.

---

## 2. 8-Emotion Color Taxonomy

Each of the 8 target emotions has a dedicated color identity used consistently across badges, progress bars, charts, and indicators:

| Emotion Label | Hex Code | Tailwind Class | Semantic Rationale |
| :--- | :--- | :--- | :--- |
| **`happy`** | `#10B981` | `bg-emerald-500` | Warm, positive green representing customer delight. |
| **`sad`** | `#0284C7` | `bg-sky-600` | Cool blue representing disappointment or sadness. |
| **`angry`** | `#EF4444` | `bg-red-500` | Bright crimson red representing high-severity anger. |
| **`frustrated`** | `#F97316` | `bg-orange-500` | Burnt orange representing friction and frustration. |
| **`surprised`** | `#8B5CF6` | `bg-purple-500` | Vibrant violet representing surprise or unexpected results. |
| **`fearful`** | `#6366F1` | `bg-indigo-500` | Deep indigo representing anxiety or concern. |
| **`disgusted`** | `#84CC16` | `bg-lime-500` | Olive / yellow-green representing aversion or disgust. |
| **`neutral`** | `#64748B` | `bg-slate-500` | Muted slate gray representing factual, emotionless text. |

---

## 3. Screen Layout & Component Hierarchy

The desktop interface uses a clean two-column grid layout with a top navigation bar:

```
+-----------------------------------------------------------------------------------+
|  Review Sense 🧠           [DeBERTa-v3 Active ⚡]   [API: 24ms]   [GitHub Repo ↗] |
+-----------------------------------------------------------------------------------+
|                                        |                                          |
|  LEFT PANEL: Input & Presets           |  RIGHT PANEL: Emotion Results & Analytics|
|                                        |                                          |
|  +----------------------------------+  |  +------------------------------------+  |
|  | Enter product review text...     |  |  | HERO CARD                          |  |
|  |                                  |  |  | Primary: ANGRY 🔥 (84%)             |  |
|  |                                  |  |  | Secondary: [Frustrated 52%]         |  |
|  +----------------------------------+  |  +------------------------------------+  |
|  [ Analyze Emotion ]  [ Clear ]        |  |                                      |  |
|                                        |  | PROBABILITY DISTRIBUTION (All 8)     |  |
|  Quick Presets:                        |  | Happy     [==               ] 12%    |  |
|  [ Shipping Delay ] [ Great Item ]     |  | Angry     [==============   ] 84% *  |  |
|  [ Defective Unit ] [ Mixed Feedback ] |  | Frustrated[==========       ] 52% *  |  |
|                                        |  | Neutral   [====             ] 18%    |  |
|  [ 📁 Upload Batch CSV... ]            |  +------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 4. Key Interactive Components

### 4.1 Header Bar (`Header.tsx`)
* App Branding & Logo icon.
* **Active Engine Pill:** Displays `DeBERTa-v3` (Green dot) or `TF-IDF Baseline` (Yellow dot).
* **Latency Counter:** Displays API response execution time in milliseconds (`⚡ 28ms`).

### 4.2 Review Input Panel (`ReviewInput.tsx`)
* Large multi-line textarea with auto-focus.
* Live character counter (`42 / 2000 chars`).
* Preset Chips for 1-click test scenarios (*"Broken item on arrival"*, *"Super fast shipping, love it!"*, *"Confused by setup instructions"*).

### 4.3 Hero Emotion Card (`HeroEmotionCard.tsx`)
* Large colored badge displaying the **Primary Emotion**.
* Prominent confidence percentage text (e.g. `84% Confidence`).
* Secondary emotion tags that passed decision threshold calibration.

### 4.4 Probability Bar Chart (`ProbabilityBars.tsx`)
* 8 horizontal progress bars animated via CSS transitions (`transition-all duration-500`).
* Calibrated threshold marker indicator ($\mid$) rendered on each bar so users see *why* secondary emotions were selected.

---

## 5. Responsive Design & Accessibility

* **Breakpoints:** Single-column stacked view for screens $< 768\text{px}$ (mobile), side-by-side grid for $\ge 768\text{px}$ (desktop).
* **Keyboard Navigation:** Full tab support and `Cmd+Enter` / `Ctrl+Enter` shortcut to submit review text.


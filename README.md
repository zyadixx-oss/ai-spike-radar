# AI Spike Radar — Deriv Boom & Crash

**مختبر بحثي مؤتمت** لدراسة انفجارات مؤشرات Boom & Crash على Deriv: يجمع البيانات الحقيقية، يكتشف الانفجارات، يبحث عن الأنماط التي تسبقها مقارنةً بنوافذ ضابطة، يختبر عشرات الاستراتيجيات والنماذج اختبارًا صارمًا خالٍ من التسرب، ويعرض رادارًا لحظيًا باحتمالات **معايرة**. وإن لم يجد دليلًا إحصائيًا يقول ذلك صراحة.

> ⛨ **بحث فقط.** `ready_for_live = false` · `live_allowed = false` · `opened_trades = false` · `LIVE_TRADING = false` ثوابت في الكود (`backend/app/safety.py`) وليست إعدادات. عميل Deriv يرفض قبل الإرسال أي رسالة خارج قائمة القراءة البيضاء (`ticks_history`, `ticks`, `active_symbols`, `ping`, …). لا يوجد `authorize`، ويرفض التطبيق الإقلاع إذا وُجد `DERIV_API_TOKEN` أو `LIVE_TRADING=true`.

*English summary:* an automated research lab for Deriv Boom & Crash spikes — real data collection, spike detection, pre-spike pattern research with control windows, a 23-family strategy arena, 13 ML models vs 5 reference baselines with purged walk-forward validation and a hard holdout acceptance gate, a live radar with calibrated probabilities, an append-only prediction log, forward/shadow testing and optional Telegram alerts. Research only; trading is impossible by construction.

---

## المحتويات
1. [التشغيل السريع](#التشغيل-السريع)
2. [النشر: Docker Compose / VPS / Render / Vercel](#النشر)
3. [كيف يعمل البحث (التعاريف والمنهجية)](#المنهجية)
4. [الصفحات](#الصفحات)
5. [الإعدادات](#الإعدادات)
6. [الاختبارات](#الاختبارات)
7. [القيود والملاحظات الصادقة](#القيود)
8. [بنية المشروع و API](#البنية)

---

## التشغيل السريع

المتطلبات: Python 3.11+ (مُختبر على 3.13)، Node 20+.

```bash
cd backend
pip install -r requirements-dev.txt
```

```bash
# اختياري: نماذج LSTM / Temporal CNN / Transformer (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
# 1) تحقّق من الاتصال الحقيقي بـ Deriv (قراءة فقط): يكتشف الرموز ويجلب تيكات
python -m app.cli probe
```

```bash
# 2) شغّل الخادم: API + جامع البيانات + الرادار + منفّذ المهام (SQLite افتراضيًا في backend/data/)
python -m app.cli serve --port 8000
```

```bash
# 3) الواجهة (وضع التطوير، تحوّل /api و /ws إلى المنفذ 8000)
cd ../frontend && npm install && npm run dev
```

افتح http://localhost:5173. أو ابنِ الواجهة (`npm run build`) وسيخدمها الخادم مباشرة على http://localhost:8000.

ما يحدث تلقائيًا بعد الإقلاع:
1. الاتصال بـ Deriv واكتشاف الرموز من `active_symbols` بمطابقة الاسم المعروض (`Boom 300 Index` …) — لا تُفترض الرموز البرمجية.
2. سدّ أي فجوة منذ آخر تشغيل، ثم الاشتراك اللحظي، ثم جمع التاريخ للخلف (حتى `MAX_BACKFILL_DAYS`) مع حفظ التقدم واستئنافه.
3. المجدول الدائم يضيف المهام: كشف الانفجارات (5 د)، تقييم التوقعات (30 ث)، البحث (3 س)، التدريب (6 س)، المختبر (12 س).
4. الرادار يحسب الاحتمالات كل ثانية ويسجّل التوقعات قبل معرفة نتائجها.

تشغيل مهمة يدويًا: `python -m app.cli job train <SYMBOL>` (أو من الواجهة: زر "تشغيل الآن").

---

## النشر

### Docker Compose (VPS أو محلي)
```bash
cp .env.example .env
```
```bash
docker compose up -d --build
```
- `backend` (المنفذ 8000): API + الجامع + الرادار (في عملية واحدة لأن الرادار يقرأ التيكات الحية من الذاكرة).
- `worker`: منفّذ المهام الثقيلة (تدريب/مختبر/بحث) — يتشارك قاعدة البيانات، والنماذج المدرّبة تُخزَّن داخل قاعدة البيانات (`model_artifacts`) فلا حاجة لقرص مشترك.
- `db`: PostgreSQL + TimescaleDB.
- `frontend` (المنفذ 8080): Nginx يخدم الواجهة ويحوّل `/api` و `/ws`.

لتفعيل نماذج التعلم العميق: `INSTALL_TORCH=true docker compose build`.

على VPS: ضع خلف Nginx/Caddy مع TLS، وافتح المنفذ 8080 فقط. أعِد التشغيل بأمان في أي وقت: المهام المقطوعة تعود للطابور (Lease)، والجمع يستأنف من آخر نقطة، والفجوات تُسدّ من التاريخ.

### Render (الخلفية) + Vercel (الواجهة)
1. Render → New → Blueprint → اختر المستودع؛ يقرأ `render.yaml` (خدمة web + worker + Postgres).
2. Vercel → استورد مجلد `frontend`، واضبط `VITE_API_BASE=https://<your-render-api>.onrender.com`. ملف `frontend/vercel.json` يعالج مسارات SPA.
3. اضبط `CORS_ORIGINS` في Render على نطاق Vercel.

---

## المنهجية

### تعريف الانفجار (قابل للقياس)
لكل رمز: `r_i = ln(p_i/p_{i-1})` داخل مقطع خالٍ من الفجوات؛ `σ = 1.4826·MAD(r)`؛ انفجار عند `direction·r_i > k·σ` (k=8 افتراضيًا، Boom للأعلى و Crash للأسفل). التيكات المتتالية تُدمج في حدث واحد. التصنيف صغير/متوسط/كبير بثُلثيّات حجم انفجارات الرمز نفسه. تُعرض حساسية العدد لقيم k مختلفة.

### الفجوات
فجوة = فرق زمني > max(15ث، 10×وسيط الفاصل). البيانات تُقسَّم إلى مقاطع: لا عائد ولا ميزة ولا تسمية تعبر فجوة. التوقعات التي تقع نافذتها على فجوة تُقيَّم `invalid` ولا تُحسب نجاحًا أو فشلًا.

### الميزات (سببية بالكامل)
- **Tick:** تذبذب/سرعة/تسارع/نسبة الصعود/الإنتروبيا/طول السلسلة/Z-score/RSI/التيكات والثواني منذ آخر انفجار/عدد الانفجارات الأخيرة… بنوافذ 10–2000 تيك.
- **الشموع (لكل فريم):** BB Squeeze، ATR Compression، RSI، StochRSI، MACD، ADX، Keltner، Donchian، EMA، Momentum، Price Action، Candle Compression، Liquidity Sweep، ICT (premium/discount + displacement)، CRT، FVG، Market Structure/BOS، Volatility Regime، عدد التيكات للشمعة (لا يوجد حجم تداول — لا نصطنعه).
- الشمعة تُستخدم فقط بعد إغلاقها (`avail = نهاية الشمعة`). σ والعتبة تُقدَّر من أول 60% من البيانات فقط.
- اختبار آلي يتحقق أن تغيير/حذف المستقبل لا يغيّر أي ميزة.

### التحقق
```
| DEV 60% (purged walk-forward → اختيار النموذج) |purge| VAL 20% (معايرة + حدود الألوان) |purge| TEST 20% (بوابة قبول فقط) |
```
- **Purge** بطول أفق التسمية، و**Embargo** 500 تيك بين الكتل.
- **المرجعيات:** عشوائي، المعدل الأساسي، نموذج هندسي عديم الذاكرة، Hazard الفواصل (حسب التيكات منذ آخر انفجار)، Weibull Survival.
- **النماذج:** Logistic، Random Forest، Gradient Boosting، XGBoost، LightGBM، Isolation Forest، Discrete-time Hazard (Survival)، Gaussian HMM (ترشيح أمامي سببي، لا تنعيم)، LSTM، Temporal CNN، Transformer.
- **الاختيار** على متوسط Brier Skill Score في Walk-forward خارج العينة (والأبسط عند التعادل). **الاختبار النهائي لا يُستخدم للاختيار أبدًا.**
- **القبول (approved):** على TEST — الحد الأدنى لـ 95% CI (Bootstrap كتلي) لـ BSS مقابل المعدل الأساسي > 0 **و** مقابل أفضل مرجعي (المختار على VAL) > 0 **و** مكسب PR-AUC > 0، مع ≥15 إيجابية، ووجود حد إشارة دال على VAL.
- **الاحتمالات معايرة** (Isotonic على VAL، أو Platt عند قلة الإيجابيات) مع ECE ومنحنى موثوقية.

### ألوان الرادار (مستخرجة من البيانات)
- الحدود تُبحث فقط فوق المعدل الأساسي. **أصفر:** أدنى حد يكون فيه الحد الأدنى لفاصل ويلسون للدقة > المعدل الأساسي. **برتقالي:** حد F1 الأمثل بين الحدود الدالة. **أحمر:** أدنى حد أعلى من البرتقالي يكون حده الأدنى ≥ دقة البرتقالي، **ولنموذج معتمد فقط**. **رمادي:** لا نموذج/بيانات متوقفة/فشل نموذج. نموذج غير معتمد لا يتجاوز الأصفر.
- الدلالة تُحسب على **محاولات غير متداخلة** (أول إشارة في كل نافذة بطول الأفق، مع فحص منفصل للنوافذ الزوجية والفردية) — لأن الصفوف المتجاورة تتشارك نفس الانفجار المستقبلي وتضخّم الدلالة الزائفة. اختبار تحكّم سلبي يثبت ذلك.

### مختبر الاستراتيجيات
كل قاعدة = (عائلة، فريم، ميزة، اتجاه، كمية 1/5/10/20%) أو علامة ثنائية. ~2500 قاعدة لكل أفق: تصفية على DEV ← FDR (Benjamini-Hochberg 10%) على VAL ← تقرير TEST. ثم دمج AND لأفضل 8. تداول افتراضي (Profit Factor، Max DD، صافي بعد التكلفة، عدد الصفقات) مع **مرجع "دخول في كل فرصة"** للمقارنة.

### أبحاث الأنماط
- نوافذ قبل الانفجار: 5/10/30ث، 1/3/5/10/15/30د، 50/100/200/500 تيك، مقابل نوافذ ضابطة لم يتبعها انفجار خلال 30 تيك. Mann-Whitney + FDR + **تكرار** على نصفي التاريخ.
- هل الفواصل عديمة الذاكرة؟ KS مقابل الأسّي، χ² لاستواء الـ Hazard، شكل Weibull مع CI. **لا يُفترض أن تأخر الانفجار يعني اقترابه** — يُختبر.
- دراسة حدث (متوسط المسار ±200 تيك).

### التشغيل المستمر
توقعات تُسجَّل قبل النتيجة في جدول **غير قابل للتعديل/الحذف** (Triggers في SQLite وPostgreSQL)، والنتائج في جدول منفصل insert-only. تقييم أمامي (Forward) + اختبار ظل (Shadow) للمتحدّين. البطل المعتمد لا يُستبدل إلا بنموذج أفضل **بدلالة** على نافذة حديثة لم يرها أي منهما.

---

## الصفحات
| الصفحة | المحتوى |
|---|---|
| Dashboard | حالة الاتصال، الأعداد، أقوى الإشارات، آخر الانفجارات، إجابات أسئلة البحث |
| Live Spike Radar | كل الرموز مرتبة حسب قوة الدليل لكل أفق، تفاصيل كل توقع (الاتجاه، الاحتمال، الأفق، الحجم المتوقع، الفريم الأفضل، الأسباب، النموذج، الأداء، الإنذارات الكاذبة، الصلاحية) |
| Live Charts | شموع حية لـ 15 فريمًا، EMA/Bollinger/RSI، الانفجارات، الإشارات ✓/✗، شريط الاحتمال، **Bar Replay** (النتائج لا تظهر قبل انقضاء الصلاحية) |
| Spike History | الإحصاءات، Hazard، دراسة الحدث، الأنماط مقابل النوافذ الضابطة، القائمة |
| AI Strategy Arena | أفضل قاعدة لكل عائلة، الدمج، التداول الافتراضي والمرجع |
| Backtesting Lab | IS / WF / VAL / TEST لكل نموذج، شروط القبول، المعايرة، الكشف المسبق ومدة الإنذار |
| Timeframe Rankings | ترتيب الفريمات لكل رمز وأفق |
| Model Rankings | ترتيب النماذج والمرجعيات + سجل البطل/المتحدي |
| Prediction History | السجل الكامل غير القابل للتعديل مع النتائج |
| Shadow Testing | الأداء الأمامي للأبطال والمتحدّين والمعدل الأساسي |
| System Health | الاتصال، التغطية، الفجوات، المهام، الأحداث، قيود الأمان |
| Reports | إجابات الأسئلة السبعة لكل رمز + تنزيل JSON |

الواجهة عربية/إنجليزية (RTL/LTR)، داكنة، متجاوبة مع الجوال.

---

## الإعدادات
انظر `.env.example`. أهمها: `DATABASE_URL`، `DERIV_APP_ID`، `DERIV_WS_URL`، `DERIV_WS_PROXY`، `MAX_BACKFILL_DAYS`، `SPIKE_K`، `FEATURE_TIMEFRAMES`، `MAX_TRAIN_ROWS`، `SIM_COST_PER_TRADE_PCT`، فترات الجدولة، `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`، `ALERT_MIN_LEVEL` (افتراضي orange — التنبيه فقط لنموذج معتمد، مع منع التكرار لنفس الرمز/الأفق/النافذة).

---

## الاختبارات
```bash
cd backend && python -m pytest -q
```
| الملف | ما يثبته |
|---|---|
| `test_safety.py` | الثوابت false، رفض buy/sell/proposal/authorize/أي رسالة مجهولة قبل الإرسال، رفض الإقلاع مع LIVE_TRADING أو توكن، `/api/trade` = 403، فحص المصدر |
| `test_leakage.py` | الميزات لا تتغير بتغيير/حذف المستقبل، التسميات تحترم الفجوات، عدم تداخل التقسيمات، σ من بيانات التدريب فقط |
| `test_core.py` | الشموع، دقة كشف الانفجارات، عدم العبور فوق الفجوات، الاكتشاف بالاسم، المقاييس، **تحكّم سلبي لاختبار الدلالة** |
| `test_persistence.py` | منع UPDATE/DELETE على التوقعات والنتائج، استئناف المهام بعد "انهيار"، إعادة المحاولة، منع التكرار، حالة المجدول |
| `test_deriv_client.py` | خادم وهمي بنفس البروتوكول: اكتشاف، Backfill مستأنف، شموع، بث لحظي، **قطع الاتصال ← إعادة الاتصال ← سدّ الفجوة** حتى تصبح البيانات متصلة، وإثبات أن لا رسالة تداول أُرسلت |
| `test_pipeline_controls.py` | **تحكّم سلبي:** على بيانات عديمة الذاكرة لا يُعتمد أي نموذج ولا تتأكد أي استراتيجية. **تحكّم إيجابي:** نمط مزروع يُكتشف ويُعتمد |
| `test_radar_e2e.py` | من التيكات إلى التدريب إلى الرادار إلى التسجيل والتقييم والتنبيهات، فشل النموذج ← رمادي، البيانات المتوقفة ← رمادي، استبدال البطل فقط بتحسن مثبت |
| `test_api.py` | كل نقاط API تعمل، والحالات الفارغة صادقة |

بيانات الاختبار مولّدة في `tests/sim.py` **للاختبار فقط** ولا تدخل قاعدة بيانات التطبيق أبدًا.

---

## القيود

- **الاتصال الحقيقي بـ Deriv لم يُتحقق منه داخل بيئة البناء**: وكيل الشبكة في تلك البيئة لا يدعم ترقية WebSocket (Cloudflare أعاد HTTP 520). العميل مُختبر بالكامل ضد خادم بنفس البروتوكول. **شغّل `python -m app.cli probe` على خادمك أولًا**؛ إذا تغيّر عنوان API لدى Deriv عدّل `DERIV_WS_URL`.
- عمق تاريخ التيكات عبر API غير موثّق؛ يتراكم التاريخ مع التشغيل المستمر.
- تصف Deriv هذه المؤشرات بأنها مُولّدة بمتوسط تكرار ثابت؛ إن كانت عديمة الذاكرة فعلًا فالنتيجة المتوقعة الصادقة هي: **لا نموذج معتمد ولا لون فوق الأخضر/الأصفر**. هذا ليس عطلًا.
- تكلفة التداول الافتراضي افتراض قابل للضبط؛ لا تُفسَّر نتائج التداول الافتراضي كأرباح متوقعة.
- مدة الإنذار ونسبة الكشف المسبق على TEST حدود دنيا لأن صفوف التقييم مأخوذة بعيّنة كل N تيك.
- نماذج التعلم العميق تعمل فقط مع PyTorch وبيانات كافية (≥5000 صف و≥100 إيجابية)، وإلا تُسجَّل "skipped" مع السبب.

---

## البنية
```
backend/app
  safety.py            ثوابت الأمان + قائمة رسائل Deriv المسموحة
  config.py, db.py     الإعدادات، المخطط، Triggers عدم التعديل
  deriv/               client (إعادة اتصال، حد معدل)، discovery، collector (backfill/live/gaps)
  research/            data, candles, spikes, features, labels, validation, metrics, models, deep,
                       dataset, train (pipeline + registry), arena, patterns, trading (افتراضي), run
  engine/              jobs (طابور دائم)، tasks، radar، livefeatures، evaluate، alerts
  api/routes.py, main.py, cli.py
frontend/src           React + Vite + TS + Tailwind + lightweight-charts; 12 صفحة؛ i18n عربي/إنجليزي
docs/ACCEPTANCE_CHECKLIST.md   قائمة القبول، الافتراضات، المصادر، حالات الفشل، الخطة
```
أهم نقاط API: `/api/radar`، `/api/radar/{symbol}`، `/api/candles`، `/api/markers`، `/api/spikes`، `/api/research/{symbol}`، `/api/arena`، `/api/backtests`، `/api/rankings/models`، `/api/rankings/timeframes`، `/api/predictions`، `/api/shadow`، `/api/reports/summary`، `/api/health`، `/api/jobs`، WebSocket `/ws`.

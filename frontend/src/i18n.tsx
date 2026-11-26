import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Lang = "ar" | "en";

const dict: Record<string, { ar: string; en: string }> = {
  app_title: { ar: "رادار الانفجارات الذكي", en: "AI Spike Radar" },
  app_sub: { ar: "Deriv Boom & Crash — مختبر بحثي", en: "Deriv Boom & Crash — research lab" },
  research_only: { ar: "بحث ومراقبة فقط — لا تداول حقيقي", en: "Research & monitoring only — no live trading" },
  nav_dashboard: { ar: "لوحة التحكم", en: "Dashboard" },
  nav_radar: { ar: "رادار الانفجارات", en: "Live Spike Radar" },
  nav_charts: { ar: "الشارتات الحية", en: "Live Charts" },
  nav_spikes: { ar: "سجل الانفجارات", en: "Spike History" },
  nav_arena: { ar: "مختبر الاستراتيجيات", en: "AI Strategy Arena" },
  nav_backtest: { ar: "الاختبارات التاريخية", en: "Backtesting Lab" },
  nav_tf: { ar: "ترتيب الفريمات", en: "Timeframe Rankings" },
  nav_models: { ar: "ترتيب النماذج", en: "Model Rankings" },
  nav_predictions: { ar: "سجل التوقعات", en: "Prediction History" },
  nav_shadow: { ar: "الاختبار الحي الافتراضي", en: "Shadow Testing" },
  nav_health: { ar: "حالة النظام", en: "System Health" },
  nav_reports: { ar: "التقارير", en: "Reports" },
  symbol: { ar: "المؤشر", en: "Symbol" },
  horizon: { ar: "الأفق", en: "Horizon" },
  probability: { ar: "الاحتمال", en: "Probability" },
  base_rate: { ar: "المعدل الطبيعي", en: "Base rate" },
  lift: { ar: "الرفع", en: "Lift" },
  level: { ar: "المستوى", en: "Level" },
  model: { ar: "النموذج", en: "Model" },
  direction: { ar: "الاتجاه", en: "Direction" },
  up: { ar: "صعود", en: "Up" },
  down: { ar: "هبوط", en: "Down" },
  approved: { ar: "معتمد", en: "Approved" },
  not_approved: { ar: "غير معتمد", en: "Not approved" },
  reasons: { ar: "أسباب الإشارة", en: "Signal reasons" },
  expires: { ar: "انتهاء الصلاحية", en: "Expires" },
  expected_size: { ar: "الحجم المتوقع", en: "Expected size" },
  best_tf: { ar: "الفريم الأفضل", en: "Best timeframe" },
  perf: { ar: "الأداء التاريخي", en: "Historical performance" },
  false_alarms: { ar: "الإنذارات الكاذبة", en: "False alarms" },
  lvl_gray: { ar: "لا توجد أدلة كافية", en: "Insufficient evidence" },
  lvl_green: { ar: "مراقبة عادية", en: "Normal monitoring" },
  lvl_yellow: { ar: "نمط يستحق المراقبة", en: "Pattern worth watching" },
  lvl_orange: { ar: "احتمال أعلى من الطبيعي", en: "Above-normal probability" },
  lvl_red: { ar: "إشارة مرتفعة (نموذج معتمد)", en: "High signal (approved model)" },
  loading: { ar: "جارٍ التحميل…", en: "Loading…" },
  no_data: { ar: "لا توجد بيانات بعد", en: "No data yet" },
  error: { ar: "خطأ", en: "Error" },
  live: { ar: "مباشر", en: "Live" },
  stale: { ar: "متوقف", en: "Stale" },
  connected: { ar: "متصل", en: "Connected" },
  disconnected: { ar: "غير متصل", en: "Disconnected" },
  ticks: { ar: "تيكات", en: "Ticks" },
  spikes: { ar: "انفجارات", en: "Spikes" },
  predictions: { ar: "توقعات", en: "Predictions" },
  outcomes: { ar: "نتائج مقيّمة", en: "Evaluated outcomes" },
  hit: { ar: "صحيح", en: "Hit" },
  miss: { ar: "فاشل", en: "Miss" },
  invalid: { ar: "غير صالح", en: "Invalid" },
  pending: { ar: "قيد الانتظار", en: "Pending" },
  timeframe: { ar: "الفريم", en: "Timeframe" },
  replay: { ar: "إعادة التشغيل", en: "Bar Replay" },
  play: { ar: "تشغيل", en: "Play" },
  pause: { ar: "إيقاف", en: "Pause" },
  step: { ar: "خطوة", en: "Step" },
  exit_replay: { ar: "خروج من الإعادة", en: "Exit replay" },
  indicators: { ar: "المؤشرات", en: "Indicators" },
  run_now: { ar: "تشغيل الآن", en: "Run now" },
  verdict: { ar: "الخلاصة", en: "Verdict" },
  memoryless: { ar: "عديم الذاكرة؟", en: "Memoryless?" },
  yes: { ar: "نعم", en: "Yes" },
  no: { ar: "لا", en: "No" },
  all: { ar: "الكل", en: "All" },
  time: { ar: "الوقت", en: "Time" },
  size: { ar: "الحجم", en: "Size" },
  safety_flags: { ar: "قيود الأمان", en: "Safety flags" },
  disclaimer: {
    ar: "الاحتمالات تقديرات إحصائية معايرة وليست يقينًا. لا يُرسل النظام أي أوامر شراء أو بيع.",
    en: "Probabilities are calibrated statistical estimates, never certainties. The system never sends buy/sell orders.",
  },
};

type Ctx = { lang: Lang; setLang: (l: Lang) => void; t: (k: string) => string };
const I18n = createContext<Ctx>({ lang: "ar", setLang: () => {}, t: (k) => k });

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    try {
      return (localStorage.getItem("lang") as Lang) || "ar";
    } catch {
      return "ar";
    }
  });
  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    try {
      localStorage.setItem("lang", lang);
    } catch {
      /* ignore */
    }
  }, [lang]);
  const t = (k: string) => dict[k]?.[lang] ?? k;
  return <I18n.Provider value={{ lang, setLang, t }}>{children}</I18n.Provider>;
}

export const useI18n = () => useContext(I18n);

/** Inline bilingual text for page-specific copy. */
export function useTx() {
  const { lang } = useI18n();
  return (ar: string, en: string) => (lang === "ar" ? ar : en);
}
